import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm

from config import InferenceConfig
from ..dataset.data_utils import get_groups, BEAT_TYPES
from ..tokenizer import ContextType, EventType, Event
from .preprocessor import Preprocessor
from .processor import Processor, GenerationConfig, MILISECONDS_PER_SECOND
from ..tokenizer import MILISECONDS_PER_STEP


class SuperTimingGenerator:
    def __init__(
            self,
            args: InferenceConfig,
            model,
            tokenizer,
    ):
        self.args = args
        self.model = model
        self.preprocessor = Preprocessor(args, parallel=True)
        self.processor = Processor(args, model, tokenizer, cfg_scale=args.timer_cfg_scale)
        self.processor.do_sample = False
        self.processor.parallel = True
        self.processor.num_beams = args.timer_num_beams
        self.processor.top_p = 1
        self.processor.top_k = 50
        self.bpm_change_threshold = args.timer_bpm_threshold
        self.types_first = args.osut5.data.types_first
        self.iterations = args.timer_iterations

        self.frame_seq_len = args.osut5.data.src_seq_len - 1
        self.frame_size = args.osut5.model.spectrogram.hop_length
        self.sample_rate = args.osut5.model.spectrogram.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.miliseconds_per_sequence = self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate

    def generate(
            self,
            audio: npt.ArrayLike,
            generation_config: GenerationConfig,
            verbose: bool = False,
    ):
        # Prepare beat histograms
        num_miliseconds = len(audio) * MILISECONDS_PER_SECOND // self.sample_rate
        beats_hist = np.zeros([num_miliseconds], dtype=int)
        measures_hist = np.zeros([num_miliseconds], dtype=int)
        timing_points_hist = np.zeros([num_miliseconds], dtype=int)
        tpbs = []
        measure_counts = []

        if verbose:
            print("Generating timing")

        iterations = self.iterations
        iterator = tqdm(list(range(iterations))) if verbose else range(iterations)
        for _ in iterator:
            audio_offset = np.random.randint(-(self.miliseconds_per_sequence // 2), self.miliseconds_per_sequence // 2)
            begin_pad = max(0, audio_offset * self.sample_rate // MILISECONDS_PER_SECOND)
            begin_remove = max(0, -audio_offset * self.sample_rate // MILISECONDS_PER_SECOND)
            sequences = self.preprocessor.segment(audio[begin_remove:], begin_pad, 0)
            events, _ = self.processor.generate(
                sequences=sequences,
                generation_config=generation_config,
                in_context=[ContextType.NONE],
                out_context=[ContextType.MAP] if self.args.osut5.data.add_timing else [ContextType.TIMING],
                verbose=False,
            )[0]
            groups, _ = get_groups(events, types_first=self.types_first)
            last_beat_time = None
            last_group_type = None
            last_measure_time = None
            measure_counter = None
            for group in groups:
                time = group.time - audio_offset
                if time < 0 or time >= num_miliseconds:
                    continue
                if group.event_type not in BEAT_TYPES:
                    continue
                if group.event_type == EventType.BEAT:
                    beats_hist[time] += 1

                    if measure_counter is not None:
                        measure_counter += 1
                elif group.event_type == EventType.MEASURE:
                    measures_hist[time] += 1

                    if measure_counter is not None:
                        measure_counts.append((last_measure_time, measure_counter))

                    last_measure_time = time
                    measure_counter = 1
                elif group.event_type == EventType.TIMING_POINT:
                    timing_points_hist[time] += 1
                    last_measure_time = time
                    measure_counter = 1

                if (last_beat_time is not None and last_beat_time != time and
                        not (group.event_type == EventType.TIMING_POINT and last_group_type != EventType.TIMING_POINT)):
                    tpb = (time - last_beat_time) // MILISECONDS_PER_STEP
                    if 20 < tpb < 100:
                        tpbs.append((last_beat_time, tpb))

                last_beat_time = time
                last_group_type = group.event_type

        # Smooth and normalize histograms
        beats_hist = gaussian_filter1d(beats_hist.astype(float), 10) / iterations * 50
        measures_hist = gaussian_filter1d(measures_hist.astype(float), 10) / iterations * 50
        timing_points_hist = gaussian_filter1d(timing_points_hist.astype(float), 10) / iterations * 50

        # Sort the ticks per beats points
        tpbs = sorted(tpbs, key=lambda x: x[0])

        signal = beats_hist + measures_hist + timing_points_hist * 2
        peakind, properties = find_peaks(signal, distance=50, prominence=0.1, rel_height=1, width=2, wlen=50)
        prominences = properties["prominences"]

        # For each peak determine the BPM by taking nearby BPMs and get the interpolated most common BPM
        # Use peak finding and take the highest peak
        # If there is no clear peak, we don't assign a BPM value and instead infer it from the surrounding BPMs
        def get_peak_bpms(w=300, thresh=0.6):
            peak_bpms = []
            for peak in peakind:
                nearby_tpbs = [tpb for time, tpb in tpbs if peak - w < time < peak + w]
                hist, bins = np.histogram(nearby_tpbs, bins=range(20, 100))
                if hist.max() > thresh * hist.sum():
                    peak_bpms.append(60_000 / (bins[np.argmax(hist)] * 10))
                else:
                    peak_bpms.append(np.nan)

            return np.array(peak_bpms)

        peak_bpms = get_peak_bpms(200, self.bpm_change_threshold)
        peak_bpms_defined = ~np.isnan(peak_bpms)

        # Normalize BPM values to prevent parts with 2x or 0.5x the BPM
        median_bpm = 60_000 / (np.median(np.array(tpbs)[:, 1]) * 10)
        # Normalize all bpm values in to the range [bpm/1.5, bpm*1.5] by integer division or multiplication
        peak_bpms = peak_bpms / np.ceil(peak_bpms / (median_bpm * 1.5))
        peak_bpms = peak_bpms * np.ceil((median_bpm / 1.5) / peak_bpms)

        # Fill in the missing BPM values by finding the nearest BPM value
        for i, bpm in enumerate(peak_bpms):
            if not np.isnan(bpm):
                continue
            left = i - 1
            while left >= 0 and np.isnan(peak_bpms[left]):
                left -= 1
            right = i + 1
            while right < len(peak_bpms) and np.isnan(peak_bpms[right]):
                right += 1
            if left >= 0 and (right >= len(peak_bpms) or i - left <= right - i):
                peak_bpms[i] = peak_bpms[left]
            elif right < len(peak_bpms) and (left < 0 or i - left > right - i):
                peak_bpms[i] = peak_bpms[right]
            else:
                peak_bpms[i] = median_bpm

        # def test_bpm(bpm, w=1):
        #     # Generate periodic pulse train
        #     period_ms = 60_000 / bpm
        #     pulse_train = np.zeros_like(signal)
        #     for i in range(w):
        #         pulse_train[np.arange(i, len(pulse_train), period_ms).astype(int)] = 1  # Period in samples
        #     # Cross-correlation
        #     correlation = correlate(signal, pulse_train, mode="full")
        #     offset = np.argmax(correlation) - len(signal) + w // 2
        #     offset += period_ms * np.ceil(-offset / period_ms)
        #     return bpm, offset, correlation.max() / np.sqrt(bpm)
        #
        # bpm, offset, _ = max([test_bpm(bpm, 5) for bpm in np.arange(bpm - 1, bpm + 1, 0.01)], key=lambda x: x[2])

        # Go from one peak to the next. Use the BPM to estimate where the next beat should be and find the nearest.
        # Depending on how clear the beats are, stick closer to the current BPM.
        peaks = list(zip(peakind, prominences, peak_bpms, peak_bpms_defined))
        beat_times = []
        to_process: list[tuple] = sorted(peaks, key=lambda x: x[1], reverse=True)
        processed_regions: list[tuple] = []

        def remove_range(t1, t2):
            if t1 > t2:
                t1, t2 = t2, t1
            i = 0
            while i < len(to_process):
                peak = to_process[i]
                if t1 <= peak[0] <= t2:
                    to_process.pop(i)
                    i -= 1
                i += 1

        def walk(start_time, period_ms, direction):
            def loss(peak, time):
                return abs(peak[0] - time) / peak[1]

            time = start_time

            while True:
                previous_time = time
                time += direction * period_ms

                if not (0 <= time < num_miliseconds):
                    remove_range(previous_time, time)
                    break

                nearest_peak: tuple = min(peaks, key=lambda x: loss(x, time))
                if loss(nearest_peak, time) < 60:
                    time = nearest_peak[0]
                    period_ms = 60_000 / nearest_peak[2]
                else:
                    if loss(nearest_peak, time) < 300 and nearest_peak[3]:
                        # There is a beat nearby, but it's likely on another BPM
                        time -= direction * period_ms
                        break
                    # There is no beat nearby, so make an imaginary beat

                if any(t1 <= time <= t2 for t1, t2 in processed_regions):
                    # This beat has already been processed
                    break

                beat_times.append(int(time))

            # Prevent very near beats at the seams (>300 BPM)
            m = 200
            if direction > 0:
                processed_regions.append((start_time - m, time + m))
                # Remove all peaks between the previous time and the current time from to_process
                remove_range(start_time - m, time + m)
            else:
                processed_regions.append((time - m, start_time + m))
                remove_range(time - m, start_time + m)

        while to_process:
            peak = to_process.pop(0)
            time = peak[0]
            period_ms = 60_000 / peak[2]

            beat_times.append(int(time))
            walk(time, period_ms, 1)
            walk(time, period_ms, -1)

        beat_times = sorted(beat_times)

        beat_types = []
        w = 10
        for beat_time in beat_times:
            # Classify the peak as a beat, measure, or timing point
            beat = beats_hist[beat_time - w:beat_time + w].sum()
            measure = measures_hist[beat_time - w:beat_time + w].sum()
            timing_point = timing_points_hist[beat_time - w:beat_time + w].sum()
            total = beat + measure + timing_point

            if timing_point > beat and timing_point > measure and total > 1:
                # Ignore timing points with low evidence
                event_type = EventType.TIMING_POINT
            else:
                event_type = EventType.BEAT

            beat_types.append(event_type)

        # Fix issues in the timing signature
        beats = list(zip(beat_times, beat_types))
        timing_signature = int(np.median([sig for t, sig in measure_counts]))
        cooldown = 0
        for i, (beat_time, beat_type) in enumerate(beats):
            # Positive cooldown to prevent measures too close to each other
            if cooldown > 0:
                cooldown -= 1
                continue
            # Negative cooldown to prevent beats too far away from each other
            if cooldown < 0:
                cooldown += 1
                if cooldown == 0 and beat_type != EventType.TIMING_POINT:
                    beat_types[i] = EventType.MEASURE
                    cooldown = timing_signature - 1
                continue
            if beat_type == EventType.TIMING_POINT:
                continue

            # Gather evidence for measure with timing signature
            offset_scores = []
            for k in range(timing_signature):
                score = 0
                count = 0
                for j in range(-3, 4):
                    index = i + j * timing_signature + k
                    if index < 0 or index >= len(beat_times):
                        continue
                    # If there is any timing point between current beat and the target beat, ignore
                    if any(beat_types[k] == EventType.TIMING_POINT for k in np.arange(1, abs(j)) * np.sign(j)):
                        continue

                    other_time = beat_times[index]
                    measure = measures_hist[other_time - w:other_time + w].sum()
                    timing_point = timing_points_hist[other_time - w:other_time + w].sum()
                    score += measure + timing_point
                    count += 1

                offset_scores.append(0 if count == 0 else score / count)

            if np.argmax(offset_scores) == 0:
                beat_types[i] = EventType.MEASURE
                cooldown = timing_signature - 1
            else:
                beat_types[i] = EventType.BEAT
                cooldown = -np.argmax(offset_scores)

        # Convert beats to events
        beats = list(zip(beat_times, beat_types))
        events = []
        event_times = []
        for beat_time, beat_type in beats:
            if self.types_first:
                events.append(Event(beat_type))

            events.append(Event(EventType.TIME_SHIFT, beat_time))

            if not self.types_first:
                events.append(Event(beat_type))

            event_times.append(beat_time)
            event_times.append(beat_time)

        # # plot beats+measures+timing_points histograms
        # import matplotlib as mpl
        # import matplotlib.pyplot as plt
        # mpl.use('TkAgg')
        # plt.ion()
        # plt.figure(figsize=(10, 6))
        # plt.show()
        #
        # def plot2(s):
        #     plt.cla()
        #     plt.plot(s)
        #
        # def plot3(x, y, **kwargs):
        #     plt.cla()
        #     plt.scatter(x, y, **kwargs)
        #
        # def plot():
        #     plt.cla()
        #     plt.plot(signal, label='beats')
        #     plt.plot(measures_hist, label='measures')
        #     plt.plot(timing_points_hist, label='timing_points')
        #     plt.vlines(x=peakind, ymin=-prominences, ymax=0, color='b')
        #     plt.vlines(x=beat_times, ymin=-0.5, ymax=0, color='r')
        #     # plt.legend()
        #
        # plot()

        return events, event_times
