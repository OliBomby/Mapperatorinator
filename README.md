# Mapperatorinator

Try the model [here](https://colab.research.google.com/github/OliBomby/Mapperatorinator/blob/main/colab/mapperatorinator_inference.ipynb). Check out a video showcase [here](https://youtu.be/FEr7t1L2EoA).

Mapperatorinator is multi-model framework that uses spectrogram inputs to generate fully featured osu! beatmaps for all gamemodes. The goal of this project is to automatically generate rankable quality osu! beatmaps from any song with a high degree of customizability.

This project is built upon [osuT5](https://github.com/gyataro/osuT5) and [osu-diffusion](https://github.com/OliBomby/osu-diffusion). In developing this, I spent about 2500 hours of GPU compute across 142 runs on my 4060 Ti and rented 4090 instances on vast.ai.

#### Use this tool responsibly. Always disclose the use of AI in your beatmaps.

## Installation

The instruction below allows you to generate beatmaps on your local machine, alternatively you can run it in the cloud with the [colab notebook](https://colab.research.google.com/github/OliBomby/Mapperatorinator/blob/main/colab/mapperatorinator_inference.ipynb).

### 1. Clone the repository

```sh
git clone https://github.com/OliBomby/Mapperatorinator.git
cd Mapperatorinator
```

### 2. (Optional) Create virtual environment

Use Python 3.10, later versions will might not be compatible with the dependencies.

```sh
python -m venv .venv

# In cmd.exe
.venv\Scripts\activate.bat
# In PowerShell
.venv\Scripts\Activate.ps1
# In Linux or MacOS
source .venv/bin/activate
```

### 3. Install dependencies

- Python 3.10
- [Git](https://git-scm.com/downloads)
- [ffmpeg](http://www.ffmpeg.org/)
- [PyTorch](https://pytorch.org/get-started/locally/): Make sure to follow the Get Started guide so you install `torch` and `torchaudio` with GPU support.

- and the remaining Python dependencies:

```sh
pip install -r requirements.txt
```

## Web GUI (Recommended)

For a more user-friendly experience, consider using the Web UI. It provides a graphical interface to configure generation parameters, start the process, and monitor the output.

### Launch the GUI

Navigate to the cloned `Mapperatorinator` directory in your terminal and run:

```sh
python web-ui.py
```

This will start a local web server and automatically open the UI in a new window.

### Using the GUI

- **Configure:** Set input/output paths using the form fields and "Browse" buttons. Adjust generation parameters like gamemode, difficulty, style (year, mapper ID, descriptors), timing, specific features (hitsounds, super timing), and more, mirroring the command-line options. (Note: If you provide a `beatmap_path`, the UI will automatically determine the `audio_path` and `output_path` from it, so you can leave those fields blank)
- **Start:** Click the "Start Inference" button to begin the beatmap generation.
- **Cancel:** You can stop the ongoing process using the "Cancel Inference" button.
- **Open Output:** Once finished, use the "Open Output Folder" button for quick access to the generated files.

The Web UI acts as a convenient wrapper around the `inference.py` script. For advanced options or troubleshooting, refer to the command-line instructions.

![python_u3zyW0S3Vs](https://github.com/user-attachments/assets/5312a45f-d51c-4b37-9389-da3258ddd0a1)

## Command-Line Inference

For users who prefer the command line or need access to advanced configurations, follow the steps below. **Note:** For a simpler graphical interface, please see the [Web UI (Recommended)](#web-ui-recommended) section above.

Run `inference.py` and pass in some arguments to generate beatmaps. For this use [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). See `configs/inference_v29.yaml` for all available parameters.
```
python inference.py \
  audio_path           [Path to input audio] \
  output_path          [Path to output directory] \
  beatmap_path         [Path to .osu file to autofill metadata, and output_path, or use as reference] \
  
  gamemode             [Game mode to generate 0=std, 1=taiko, 2=ctb, 3=mania] \
  difficulty           [Difficulty star rating to generate] \
  mapper_id            [Mapper user ID for style] \
  year                 [Upload year to simulate] \
  hitsounded           [Whether to add hitsounds] \
  slider_multiplier    [Slider velocity multiplier] \
  circle_size          [Circle size] \
  keycount             [Key count for mania] \
  hold_note_ratio      [Hold note ratio for mania 0-1] \
  scroll_speed_ratio   [Scroll speed ratio for mania and ctb 0-1] \
  descriptors          [List of beatmap user tags for style] \
  negative_descriptors [List of beatmap user tags for classifier-free guidance] \
  
  add_to_beatmap       [Whether to add generated content to the reference beatmap instead of making a new beatmap] \
  start_time           [Generation start time in milliseconds] \
  end_time             [Generation end time in milliseconds] \
  in_context           [List of additional context to provide to the model [NONE,TIMING,KIAI,MAP,GD,NO_HS]] \
  output_type          [List of content types to generate] \
  cfg_scale            [Scale of the classifier-free guidance] \
  super_timing         [Whether to use slow accurate variable BPM timing generator] \
  seed                 [Random seed for generation] \
```

Example:
```
python inference.py beatmap_path="'C:\Users\USER\AppData\Local\osu!\Songs\1 Kenji Ninuma - DISCO PRINCE\Kenji Ninuma - DISCOPRINCE (peppy) [Normal].osu'" gamemode=0 difficulty=5.5 year=2023 descriptors="['jump aim','clean']" in_context=[TIMING,KIAI]
```

## Interactive CLI
For those who prefer a terminal-based workflow but want a guided setup, the interactive CLI script is an excellent alternative to the Web UI.

### Launch the CLI
Navigate to the cloned directory. You may need to make the script executable first.

```sh
# Make the script executable (only needs to be done once)
chmod +x cli.sh
```

```sh
# Run the script
./cli.sh
```

### Using the CLI
The script will walk you through a series of prompts to configure all generation parameters, just like the Web UI.

It uses a color-coded interface for clarity.
It provides an advanced multi-select menu for choosing style descriptors using your arrow keys and spacebar.
After you've answered all the questions, it will display the final command for your review.
You can then confirm to execute it directly or cancel and copy the command for manual use.

## Generation Tips

- You can edit `configs/inference_v29.yaml` and add your arguments there instead of typing them in the terminal every time.
- All available descriptors can be found [here](https://osu.ppy.sh/wiki/en/Beatmap/Beatmap_tags).
- Always provide a year argument between 2007 and 2023. If you leave it unknown, the model might generate with an inconsistent style.
- Always provide a difficulty argument. If you leave it unknown, the model might generate with an inconsistent difficulty.
- Increase the `cfg_scale` parameter to increase the effectiveness of the `mapper_id` and `descriptors` arguments.
- You can use the `negative_descriptors` argument to guide the model away from certain styles. This only works when `cfg_scale > 1`. Make sure the number of negative descriptors is equal to the number of descriptors.
- If your song style and desired beatmap style don't match well, the model might not follow your directions. For example, its hard to generate a high SR, high SV beatmap for a calm song. 
- If you already have timing and kiai times done for a song, then you can give this to the model to greatly increase inference speed and accuracy: Use the `beatmap_path` and `in_context=[TIMING,KIAI]` arguments.
- To remap just a part of your beatmap, use the `beatmap_path`, `start_time`, `end_time`, and `add_to_beatmap=true` arguments.
- To generate a guest difficulty for a beatmap, use the `beatmap_path` and `in_context=[GD,TIMING,KIAI]` arguments.
- To generate hitsounds for a beatmap, use the `beatmap_path` and `in_context=[NO_HS,TIMING,KIAI]` arguments.
- To generate only timing for a song, use the `super_timing=true` and `output_type=[TIMING]` arguments.

## Overview

### Tokenization

Mapperatorinator converts osu! beatmaps into an intermediate event representation that can be directly converted to and from tokens.
It includes hit objects, hitsounds, slider velocities, new combos, timing points, kiai times, and taiko/mania scroll speeds.

Here is a small examle of the tokenization process:

![mapperatorinator_parser](https://github.com/user-attachments/assets/84efde76-4c27-48a1-b8ce-beceddd9e695)

To save on vocabulary size, time events are quantized to 10ms intervals and position coordinates are quantized to 32 pixel grid points.

### Model architecture
The model is basically a wrapper around the [HF Transformers Whisper](https://huggingface.co/docs/transformers/en/model_doc/whisper#transformers.WhisperForConditionalGeneration) model, with custom input embeddings and loss function.
Model size amounts to 219M parameters.
This model was found to be faster and more accurate than T5 for this task.

The high-level overview of the model's input-output is as follows:

![Picture2](https://user-images.githubusercontent.com/28675590/201044116-1384ad72-c540-44db-a285-7319dd01caad.svg)

The model uses Mel spectrogram frames as encoder input, with one frame per input position. The model decoder output at each step is a softmax distribution over a discrete, predefined, vocabulary of events. Outputs are sparse, events are only needed when a hit-object occurs, instead of annotating every single audio frame.

### Multitask training format

![Multitask training format](https://github.com/user-attachments/assets/62f490bc-a567-4671-a7ce-dbcc5f9cd6d9)

Before the SOS token are additional tokens that facilitate conditional generation. These tokens include the gamemode, difficulty, mapper ID, year, and other metadata.
During training, these tokens do not have accompanying labels, so they are never output by the model. 
Also during training there is a random chance that a metadata token gets replaced by an 'unknown' token, so during inference we can use these 'unknown' tokens to reduce the amount of metadata we have to give to the model.

### Seamless long generation

The context length of the model is 8.192 seconds long. This is obviously not enough to generate a full beatmap, so we have to split the song into multiple windows and generate the beatmap in small parts.
To make sure that the generated beatmap does not have noticeable seams in between windows, we use a 90% overlap and generate the windows sequentially.
Each generation window except the first starts with the decoder pre-filled up to 50% of the generation window with tokens from the previous windows.
We use a logit processor to make sure that the model can't generate time tokens that are in the first 50% of the generation window.
Additionally, the last 40% of the generation window is reserved for the next window. Any generated time tokens in that range are treated as EOS tokens.
This ensures that each generated token is conditioned on at least 4 seconds of previous tokens and 3.3 seconds of future audio to anticipate.

To prevent offset drifting during long generation, random offsets have been added to time events in the decoder during training.
This forces it to correct timing errors by listening to the onsets in the audio instead, and results in a consistently accurate offset.

### Refined coordinates with diffusion

Position coordinates generated by the decoder are quantized to 32 pixel grid points, so afterward we use diffusion to denoise the coordinates to the final positions.
For this we trained a modified version of [osu-diffusion](https://github.com/OliBomby/osu-diffusion) that is specialized to only the last 10% of the noise schedule, and accepts the more advanced metadata tokens that Mapperatorinator uses for conditional generation.

Since the Mapperatorinator model outputs the SV of sliders, the required length of the slider is fixed regardless of the shape of the control point path.
Therefore, we try to guide the diffusion process to create coordinates that fit the required slider lengths.
We do this by recalculating the slider end positions after every step of the diffusion process based on the required length and the current control point path.
This means that the diffusion process does not have direct control over the slider end positions, but it can still influence them by changing the control point path.

### Post-processing

Mapperatorinator does some extra post-processing to improve the quality of the generated beatmap:

- Refine position coordinates with diffusion.
- Resnap time events to the nearest tick using the snap divisors generated by the model.
- Snap near-perfect positional overlaps.
- Convert mania column events to X coordinates.
- Generate slider paths for taiko drumrolls.
- Fix big discrepancies in required slider length and control point path length.

### Super timing generator

Super timing generator is an algorithm that improves the precision and accuracy of generated timing by infering timing for the whole song 20 times and averaging the results.
This is useful for songs with variable BPM, or songs with BPM changes. The result is almost perfect with only sometimes a section that needs manual adjustment.

## Training

The instruction below creates a training environment on your local machine.

### 1. Clone the repository

```sh
git clone https://github.com/OliBomby/Mapperatorinator.git
cd Mapperatorinator
```

### 2. Create dataset

Create your own dataset using the [Mapperator console app](https://github.com/mappingtools/Mapperator/blob/master/README.md#create-a-high-quality-dataset). It requires an [osu! OAuth client token](https://osu.ppy.sh/home/account/edit) to verify beatmaps and get additional metadata. Place the dataset in a `datasets` directory next to the `Mapperatorinator` directory.

```sh
Mapperator.ConsoleApp.exe dataset2 -t "/Mapperatorinator/datasets/beatmap_descriptors.csv" -i "path/to/osz/files" -o "/datasets/cool_dataset"
```

### 3. Create docker container
Training in your venv is also possible, but we recommend using Docker on WSL for better performance.
```sh
docker compose up -d --force-recreate
docker attach mapperatorinator_space
```

### 4. Configure parameters and begin training

All configurations are located in `./configs/osut5/train.yaml`. Begin training by calling `osuT5/train.py`.

```sh
python osuT5/train.py -cn train_v29 train_dataset_path="/workspace/datasets/cool_dataset" test_dataset_path="/workspace/datasets/cool_dataset" train_dataset_end=90 test_dataset_start=90 test_dataset_end=100
```

## See also
- [Mapper Classifier](./classifier/README.md)
- [RComplexion](./rcomplexion/README.md)

## Credits

Special thanks to:
1. The authors of [osuT5](https://github.com/gyataro/osuT5) for their training code.
2. Hugging Face team for their [tools](https://huggingface.co/docs/transformers/index).
3. [Jason Won](https://github.com/jaswon) and [Richard Nagyfi](https://github.com/sedthh) for bouncing ideas.
4. [Marvin](https://github.com/minetoblend) for donating training credits.
5. The osu! community for the beatmaps.

## Related works

1. [osu! Beatmap Generator](https://github.com/Syps/osu_beatmap_generator) by Syps (Nick Sypteras)
2. [osumapper](https://github.com/kotritrona/osumapper) by kotritrona, jyvden, Yoyolick (Ryan Zmuda)
3. [osu-diffusion](https://github.com/OliBomby/osu-diffusion) by OliBomby (Olivier Schipper), NiceAesth (Andrei Baciu)
4. [osuT5](https://github.com/gyataro/osuT5) by gyataro (Xiwen Teoh)
5. [Beat Learning](https://github.com/sedthh/BeatLearning) by sedthh (Richard Nagyfi)
6. [osu!dreamer](https://github.com/jaswon/osu-dreamer) by jaswon (Jason Won)
