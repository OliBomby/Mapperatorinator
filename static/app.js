$(document).ready(function () {
    // Application state and configuration
    const AppState = {
        evtSource: null,
        isCancelled: false,
        inferenceErrorOccurred: false,
        accumulatedErrorMessages: [],
        errorLogFilePath: null,
        animationSpeed: 300,

        modelCapabilities: {
            "v28": {},
            "v29": {},
            "v30": {
                supportedGamemodes: ['0'],
                supportsYear: false,
                supportedInContextOptions: ['TIMING'],
                hideHitsoundsOption: true,
                supportsDescriptors: false,
            },
        }
    };

    // Utility functions
    const Utils = {
        showFlashMessage(message, type = 'success') {
            const flashContainer = $('#flash-container');
            const alertClass = type === 'success' ? 'alert success' :
                type === 'cancel-success' ? 'alert alert-cancel-success' :
                    'alert error';
            const messageDiv = $(`<div class="${alertClass}">${message}</div>`);
            flashContainer.append(messageDiv);
            setTimeout(() => messageDiv.remove(), 5000);
        },

        smoothScroll(target, offset = 0) {
            $('html, body').animate({
                scrollTop: $(target).offset().top + offset
            }, 500);
        },

        resetFormToDefaults() {
            $('#inferenceForm')[0].reset();

            // Set specific defaults
            const defaults = {
                model: 'v30', gamemode: '0', difficulty: '5', hp_drain_rate: '5',
                circle_size: '4', keycount: '4', overall_difficulty: '8',
                approach_rate: '9', slider_multiplier: '1.4', slider_tick_rate: '1',
                year: '2023', cfg_scale: '1.0', temperature: '0.9', top_p: '0.9'
            };

            Object.entries(defaults).forEach(([id, value]) => {
                $(`#${id}`).val(value);
            });

            // Reset checkboxes
            $('#hitsounded').prop('checked', true);
            $('#export_osz, #add_to_beatmap, #overwrite_reference_beatmap, #super_timing, #enable_bf16').prop('checked', false);

            // Clear descriptors and context options
            $('input[name="descriptors"], input[name="in_context_options"]')
                .removeClass('positive-check negative-check').prop('checked', false);

            // Clear paths and optional fields
            $('#audio_path, #output_path, #beatmap_path, #lora_path, #mapper_id, #seed, #start_time, #end_time, #hold_note_ratio, #scroll_speed_ratio').val('');
            PathManager.clearPlaceholders();
            PathManager.validateAndAutofillPaths(false);
        },

        showConfirmDialog(title, message, onConfirm, onCancel) {
            // Remove existing confirm dialog if any
            $('#confirm-dialog-modal').remove();

            const modalHtml = `
                <div id="confirm-dialog-modal" class="confirm-modal-overlay">
                    <div class="confirm-modal">
                        <div class="confirm-modal-header">
                            <h3>${title}</h3>
                        </div>
                        <div class="confirm-modal-body">
                            ${message}
                        </div>
                        <div class="confirm-modal-footer">
                            <button type="button" class="confirm-no">Cancel</button>
                            <button type="button" class="confirm-yes">Confirm</button>
                        </div>
                    </div>
                </div>
            `;

            $('body').append(modalHtml);

            // Handle confirm
            $('#confirm-dialog-modal .confirm-yes').on('click', () => {
                $('#confirm-dialog-modal').remove();
                if (onConfirm) onConfirm();
            });

            // Handle cancel
            $('#confirm-dialog-modal .confirm-no').on('click', () => {
                $('#confirm-dialog-modal').remove();
                if (onCancel) onCancel();
            });

            // Handle clicking outside
            $('#confirm-dialog-modal.confirm-modal-overlay').on('click', (e) => {
                if (e.target === e.currentTarget) {
                    $('#confirm-dialog-modal').remove();
                    if (onCancel) onCancel();
                }
            });

            // Prevent clicks inside modal from closing
            $('#confirm-dialog-modal .confirm-modal').on('click', (e) => e.stopPropagation());
        }
    };

    // UI Manager for conditional visibility
    const UIManager = {
        updateConditionalFields() {
            const selectedGamemode = $("#gamemode").val();
            const selectedModel = $("#model").val();
            const beatmapPath = $('#beatmap_path').val().trim();

            // Handle gamemode-based visibility
            $('.conditional-field[data-show-for-gamemode]').each(function () {
                const $field = $(this);
                const supportedModes = $field.data('show-for-gamemode').toString().split(',');
                const shouldShow = supportedModes.includes(selectedGamemode);

                if (shouldShow && !$field.is(':visible')) {
                    $field.slideDown(AppState.animationSpeed);
                } else if (!shouldShow && $field.is(':visible')) {
                    $field.slideUp(AppState.animationSpeed);
                }
            });

            // Handle model-based visibility
            $('.conditional-field[data-hide-for-model]').each(function () {
                const $field = $(this);
                const hiddenModels = $field.data('hide-for-model').toString().split(',');
                const shouldHide = hiddenModels.includes(selectedModel);

                if (shouldHide && $field.is(':visible')) {
                    $field.slideUp(AppState.animationSpeed);
                } else if (!shouldHide && !$field.is(':visible')) {
                    $field.slideDown(AppState.animationSpeed);
                }
            });

            // Handle beatmap path dependent fields
            const shouldShowBeatmapFields = beatmapPath !== '';
            ['#in-context-options-box', '#add-to-beatmap-option', '#overwrite-reference-beatmap-option'].forEach(selector => {
                const $element = $(selector);
                if (shouldShowBeatmapFields && !$element.is(':visible')) {
                    $element.fadeIn(AppState.animationSpeed);
                } else if (!shouldShowBeatmapFields && $element.is(':visible')) {
                    $element.fadeOut(AppState.animationSpeed);
                    if (selector === '#add-to-beatmap-option') {
                        $('#add_to_beatmap').prop('checked', false);
                    }
                    if (selector === '#overwrite-reference-beatmap-option') {
                        $('#overwrite_reference_beatmap').prop('checked', false);
                    }
                }
            });
        },

        updateModelSettings() {
            const selectedModel = $("#model").val();
            const capabilities = AppState.modelCapabilities[selectedModel] || {};

            // Handle gamemode restrictions
            const $gamemodeSelect = $("#gamemode");
            if (selectedModel === "v30") {
                $gamemodeSelect.val('0').prop('disabled', true);
                $gamemodeSelect.find("option").each(function () {
                    $(this).prop('disabled', $(this).val() !== '0');
                });
            } else {
                $gamemodeSelect.prop('disabled', false);
                $gamemodeSelect.find("option").prop('disabled', false);
            }

            // Handle in-context options
            const supportedContext = capabilities.supportedInContextOptions ||
                ['NONE', 'TIMING', 'KIAI', 'MAP', 'GD', 'NO_HS'];

            $('input[name="in_context_options"]').each(function () {
                const $checkbox = $(this);
                const value = $checkbox.val();
                const $item = $checkbox.closest('.context-option-item');
                const isSupported = supportedContext.includes(value);

                $item.data('model-allowed', isSupported);
                $checkbox.prop('disabled', !isSupported);

                if (isSupported) {
                    $item.slideDown(AppState.animationSpeed);
                } else {
                    $item.slideUp(AppState.animationSpeed);
                }
            });

            // Handle hitsounds for V30
            if (capabilities.hideHitsoundsOption) {
                $('#hitsounded').prop('checked', true);
            }

            // Immediately hide/show model-specific fields (Year, Descriptors)
            // This ensures they're properly hidden on page load without animation issues
            $('.conditional-field[data-hide-for-model]').each(function () {
                const $field = $(this);
                const hiddenModels = $field.data('hide-for-model').toString().split(',');
                const shouldHide = hiddenModels.includes(selectedModel);

                if (shouldHide) {
                    $field.hide();
                } else {
                    $field.show();
                }
            });

            this.updateConditionalFields();
        }
    };

    // File Browser Manager
    const FileBrowser = {
        init() {
            this.attachBrowseHandlers();
        },

        attachBrowseHandlers() {
            // Use event delegation on document for more robust handling
            $(document).on('click', '.browse-button[data-browse-type]', async function () {
                const browseType = $(this).data('browse-type');
                const targetId = $(this).data('target');

                if (!targetId) {
                    console.warn('Browse button missing data-target attribute');
                    return;
                }

                // Check if pywebview API is available
                if (!window.pywebview?.api) {
                    console.warn('pywebview API not available - running in browser mode');
                    Utils.showFlashMessage('File browser not available in browser mode. Please use the desktop app.', 'error');
                    return;
                }

                try {
                    let path;

                    if (browseType === 'folder') {
                        path = await window.pywebview.api.browse_folder();
                    } else {
                        let fileTypes = null;

                        if (targetId === 'beatmap_path') {
                            fileTypes = [
                                'Beatmap Files (*.osu)',
                                'All files (*.*)'
                            ];
                        } else if (targetId === 'audio_path') {
                            fileTypes = [
                                // todo: add more formats if needed and implement this in backend as well + add error msgs
                                'Audio Files (*.mp3;*.wav;*.ogg;*.m4a;*.flac)',
                                'All files (*.*)'
                            ];
                        }

                        path = await window.pywebview.api.browse_file(fileTypes);
                    }

                    if (path) {
                        if (targetId === 'beatmap_path' && !path.toLowerCase().endsWith('.osu')) {
                            Utils.showFlashMessage('Please select a valid .osu file.', 'error');
                            // Set the path and let validation handle inline error
                        }

                        const $targetInput = $(`#${targetId}`);
                        $targetInput.val(path);
                        console.log(`Selected ${browseType}:`, path);

                        // Trigger input event to update clear buttons and validate
                        $targetInput.trigger('input');
                        $targetInput.trigger('blur'); // Trigger blur to validate
                    }
                } catch (error) {
                    console.error(`Error browsing for ${browseType}:`, error);
                    alert(`Could not browse for ${browseType}. Ensure the backend API is running.`);
                }
            });
        }
    };

    // Path Manager for autofill, validation and clear button support
    const PathManager = {
        init() {
            this.attachPathChangeHandlers();
            this.attachClearButtonHandlers();
            $('#audio_path, #beatmap_path, #output_path, #lora_path, #background_path').trigger('blur');
        },

        attachPathChangeHandlers() {
            // Listen for input events (typing)
            $('#audio_path, #beatmap_path, #output_path, #lora_path, #background_path').on('input', (e) => {
                this.updateClearButtonVisibility(e.target);
            });

            // Listen for blur events (leaving field) - immediate validation
            $('#audio_path, #beatmap_path, #output_path, #lora_path, #background_path').on('blur', (e) => {
                this.updateClearButtonVisibility(e.target);
                this.validateAndAutofillPaths(false);
            });
        },

        attachClearButtonHandlers() {
            // Handle clear button clicks
            $('.clear-input-btn').on('click', (e) => {
                const targetId = $(e.target).data('target');
                const $targetInput = $(`#${targetId}`);

                $targetInput.val('');
                this.updateClearButtonVisibility($targetInput[0]);

                this.validateAndAutofillPaths(false);
            });

            // Initial visibility check for all fields
            $('#audio_path, #beatmap_path, #output_path, #lora_path, #background_path').each((index, element) => {
                this.updateClearButtonVisibility(element);
            });
        },

        updateClearButtonVisibility(inputElement) {
            const $input = $(inputElement);
            const $clearBtn = $input.siblings('.clear-input-btn');
            const hasValue = $input.val().trim() !== '';

            if (hasValue) {
                $clearBtn.show();
            } else {
                $clearBtn.hide();
            }
        },

        validateAndAutofillPaths(showFlashMessages = false) { // isFileDialog replaced by showFlashMessages
            const audioPath = $('#audio_path').val().trim();
            const beatmapPath = $('#beatmap_path').val().trim();
            const outputPath = $('#output_path').val().trim();

            // Only validate if at least one path is provided
            if (!audioPath && !beatmapPath && !outputPath) {
                this.clearPlaceholders();
                UIManager.updateConditionalFields();
                return Promise.resolve(true);
            }

            // Call backend validation
            return new Promise((resolve) => {
                $.ajax({
                    url: '/validate_paths',
                    method: 'POST',
                    data: {
                        audio_path: audioPath,
                        beatmap_path: beatmapPath,
                        output_path: outputPath
                    },
                    success: (response) => {
                        this.handleValidationResponse(response, showFlashMessages);
                        resolve(response.success);
                    },
                    error: (xhr, status, error) => {
                        console.error('Path validation failed:', error);
                        if (showFlashMessages) {
                            Utils.showFlashMessage('Error validating paths. Check console for details.', 'error');
                        }
                        resolve(false);
                    }
                });
            });
        },

        handleValidationResponse(response, showFlashMessages = false) {
            this.clearValidationErrors();
            const $audioPathInput = $('#audio_path');
            const $outputPathInput = $('#output_path');

            // Show autofilled paths as placeholders
            if (response.autofilled_audio_path && !$audioPathInput.val().trim()) {
                $audioPathInput.attr('placeholder', response.autofilled_audio_path);
            } else if (!$audioPathInput.val().trim()) {
                $audioPathInput.attr('placeholder', '');
            }

            if (response.autofilled_output_path && !$outputPathInput.val().trim()) {
                $outputPathInput.attr('placeholder', response.autofilled_output_path);
            } else if (!$outputPathInput.val().trim()) {
                $outputPathInput.attr('placeholder', '');
            }

            if (showFlashMessages) {
                // Show errors as flash messages and inline indicators
                response.errors.forEach(error => {
                    Utils.showFlashMessage(error, 'error');
                });
            }

            // Always show/update inline errors
            response.errors.forEach(error => {
                this.showInlineErrorForMessage(error);
            });

            // Update UI for conditional fields
            UIManager.updateConditionalFields();
        },

        showInlineErrorForMessage(error) {
            const audioPathVal = $('#audio_path').val().trim();
            const beatmapPathVal = $('#beatmap_path').val().trim();

            if (error.includes('Audio file not found') && (audioPathVal || beatmapPathVal)) {
                this.showInlineError('#audio_path', 'Audio file not found');
            } else if (error.includes('Beatmap file not found') && beatmapPathVal) {
                this.showInlineError('#beatmap_path', 'Beatmap file not found');
            } else if (error.includes('Beatmap file must have .osu extension') && beatmapPathVal) {
                this.showInlineError('#beatmap_path', 'Must be .osu file');
            }
        },

        showInlineError(inputSelector, message) {
            const $input = $(inputSelector);
            const $inputContainer = $input.closest('.input-with-clear');
            // Prevent duplicate error messages
            if ($input.siblings('.path-validation-error').length > 0) {
                $input.siblings('.path-validation-error').text(message);
            } else {
                const $errorDiv = $(`<div class="path-validation-error" style="color: #ff4444; font-size: 12px; margin-top: 2px;">${message}</div>`);
                $inputContainer.after($errorDiv);
            }
        },

        clearValidationErrors() {
            $('.path-validation-error').remove();
        },

        clearPlaceholders() {
            $('#audio_path, #output_path').attr('placeholder', '');
            this.clearValidationErrors();
        },

        // Apply placeholder values to form fields before submission
        applyPlaceholderValues() {
            const $audioPath = $('#audio_path');
            const $outputPath = $('#output_path');

            if (!$audioPath.val().trim() && $audioPath.attr('placeholder')) {
                $audioPath.val($audioPath.attr('placeholder'));
            }

            if (!$outputPath.val().trim() && $outputPath.attr('placeholder')) {
                $outputPath.val($outputPath.attr('placeholder'));
            }
        }
    };

    // Beatmap Customization Manager (Preview Time & Background)
    const BeatmapCustomization = {
        audioElement: null,
        lastAudioPath: '',
        lastPickerPosition: 0, // Store the last picker position (separate from saved preview time)

        init() {
            this.attachEventHandlers();
            this.setupBackgroundPreview();
            this.updatePreviewDisplay();
        },

        attachEventHandlers() {
            // Pick preview button
            $('#pick-preview-btn').on('click', () => this.openPreviewPicker());

            // Preview time input change
            $('#preview_time').on('input', () => this.updatePreviewDisplay());

            // Background path change - show preview
            $('#background_path').on('input blur', () => this.updateBackgroundPreview());

            // Reset preview time and background when audio path changes
            $('#audio_path').on('change blur', () => this.onAudioPathChanged());
        },

        setupBackgroundPreview() {
            // Setup clear button handler for background
            $(document).on('click', '.clear-input-btn[data-target="background_path"]', () => {
                setTimeout(() => this.updateBackgroundPreview(), 10);
            });
        },

        onAudioPathChanged() {
            const currentAudioPath = $('#audio_path').val().trim();

            // Only reset if the audio path actually changed to a different file
            if (this.lastAudioPath && currentAudioPath !== this.lastAudioPath) {
                this.clearCustomizations();
            }

            this.lastAudioPath = currentAudioPath;
        },

        clearCustomizations() {
            // Clear preview time
            $('#preview_time').val('');
            $('#preview-time-display').text('').removeClass('has-value');

            // Clear last picker position since audio changed
            this.lastPickerPosition = 0;

            // Clear background
            $('#background_path').val('');
            $('#background-preview').hide();

            // Update clear button visibility
            PathManager.updateClearButtonVisibility($('#background_path')[0]);
        },

        updatePreviewDisplay() {
            const ms = parseInt($('#preview_time').val());
            const $display = $('#preview-time-display');

            if (!isNaN(ms) && ms >= 0) {
                const seconds = ms / 1000;
                const minutes = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                $display.text(`(${minutes}:${secs.toString().padStart(2, '0')})`).addClass('has-value');
            } else {
                $display.text('').removeClass('has-value');
            }
        },

        updateBackgroundPreview() {
            const bgPath = $('#background_path').val().trim();
            const $preview = $('#background-preview');
            const $img = $('#background-preview-img');

            if (!bgPath) {
                $preview.hide();
                return;
            }

            // Request preview from backend
            $.ajax({
                url: '/get_image_preview',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ path: bgPath }),
                success: (response) => {
                    if (response.success && response.data) {
                        $img.attr('src', 'data:image/' + response.type + ';base64,' + response.data);
                        $preview.show();
                    } else {
                        $preview.hide();
                    }
                },
                error: () => {
                    $preview.hide();
                }
            });
        },

        openPreviewPicker() {
            const audioPath = $('#audio_path').val().trim() || $('#audio_path').attr('placeholder');

            if (!audioPath) {
                Utils.showFlashMessage('Please select an audio file first.', 'error');
                return;
            }

            // Create modal for preview picker
            this.createPreviewModal(audioPath);
        },

        createPreviewModal(audioPath) {
            // Remove existing modal if any
            $('#preview-picker-modal').remove();

            const modalHtml = `
                <div id="preview-picker-modal" class="preview-modal-overlay">
                    <div class="preview-modal">
                        <div class="preview-modal-header">
                            <h3>Pick Preview Point</h3>
                            <button type="button" class="preview-modal-close">×</button>
                        </div>
                        <div class="preview-modal-body">
                            <div class="preview-controls-row">
                                <label>Playback Speed:</label>
                                <select id="preview-speed">
                                    <option value="0.25">25%</option>
                                    <option value="0.5">50%</option>
                                    <option value="0.75">75%</option>
                                    <option value="1" selected>100%</option>
                                </select>
                            </div>
                            <div class="preview-slider-container" id="preview-slider-track">
                                <div class="preview-slider-fill" id="preview-slider-fill"></div>
                                <div class="preview-slider-thumb" id="preview-slider-thumb"></div>
                            </div>
                            <div class="preview-time-labels">
                                <span id="preview-current-time">0:00</span>
                                <span id="preview-total-time">--:--</span>
                            </div>
                            <div class="preview-input-row">
                                <div class="preview-input-item">
                                    <label>Milliseconds:</label>
                                    <input type="number" id="preview-ms-input" min="0" value="0" />
                                </div>
                                <div class="preview-input-item">
                                    <label>Seconds:</label>
                                    <input type="number" id="preview-sec-input" min="0" step="1" value="0" />
                                </div>
                            </div>
                            <div class="preview-volume-row">
                                <label>Volume:</label>
                                <div class="volume-slider-container" id="volume-slider-track">
                                    <div class="volume-slider-fill" id="volume-slider-fill"></div>
                                    <div class="volume-slider-thumb" id="volume-slider-thumb"></div>
                                </div>
                            </div>
                            <div class="preview-buttons-row">
                                <button type="button" id="preview-play-btn" class="browse-button">Play / Pause</button>
                                <button type="button" id="preview-test-btn" class="browse-button" title="Play 10 seconds from current position">Test Preview</button>
                                <button type="button" id="preview-set-btn" class="browse-button accent">Use This Point</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            $('body').append(modalHtml);

            // Setup audio and controls
            this.setupPreviewAudio(audioPath);
        },

        setupPreviewAudio(audioPath) {
            // Request audio data from backend
            $.ajax({
                url: '/get_audio_info',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ path: audioPath }),
                success: (response) => {
                    if (response.success) {
                        this.initializeAudioPlayer(response.url || audioPath, response.duration);
                    } else {
                        Utils.showFlashMessage('Failed to load audio: ' + (response.message || 'Unknown error'), 'error');
                        $('#preview-picker-modal').remove();
                    }
                },
                error: () => {
                    // Try direct file URL as fallback
                    this.initializeAudioPlayer('file:///' + audioPath.replace(/\\/g, '/'));
                }
            });

            // Setup modal close
            $('.preview-modal-close, .preview-modal-overlay').on('click', (e) => {
                if (e.target === e.currentTarget) {
                    this.closePreviewModal();
                }
            });

            // Prevent clicks inside modal from closing it
            $('.preview-modal').on('click', (e) => e.stopPropagation());
        },

        initializeAudioPlayer(audioUrl, duration) {
            if (this.audioElement) {
                this.audioElement.pause();
                this.audioElement = null;
            }

            this.audioElement = new Audio();
            this.audioElement.crossOrigin = 'anonymous';

            const self = this;
            const $sliderTrack = $('#preview-slider-track');
            const $sliderFill = $('#preview-slider-fill');
            const $sliderThumb = $('#preview-slider-thumb');
            const $volumeTrack = $('#volume-slider-track');
            const $volumeFill = $('#volume-slider-fill');
            const $volumeThumb = $('#volume-slider-thumb');
            const $msInput = $('#preview-ms-input');
            const $secInput = $('#preview-sec-input');
            const $currentTime = $('#preview-current-time');
            const $totalTime = $('#preview-total-time');
            const $speed = $('#preview-speed');

            // Get existing preview time to restore position, or use last picker position
            const existingPreviewTime = parseInt($('#preview_time').val()) || 0;
            const initialPosition = existingPreviewTime > 0 ? existingPreviewTime : self.lastPickerPosition;

            // Set initial volume
            this.audioElement.volume = 0.5;

            // Initialize volume slider visual (50%)
            $volumeFill.css('width', '50%');
            $volumeThumb.css('left', '50%');

            // Track state
            self.sliderDragging = false;
            self.audioLoaded = false;
            self.audioDuration = 0;
            self.currentMs = 0;

            // Function to update slider visual position
            function updateSliderVisual(ms) {
                if (self.audioDuration <= 0) return;
                const percent = Math.min(100, Math.max(0, (ms / self.audioDuration) * 100));
                $sliderFill.css('width', percent + '%');
                $sliderThumb.css('left', percent + '%');
            }

            // Function to update all displays WITHOUT changing audio position
            function updateDisplays(ms) {
                self.currentMs = ms;
                $msInput.val(Math.floor(ms));
                $secInput.val(Math.floor(ms / 1000));
                $currentTime.text(self.formatTime(ms));
                updateSliderVisual(ms);
            }

            // Simple direct seek - no event listeners, just set and verify
            function seekAudioTo(ms) {
                const targetSec = Math.max(0, Math.min(ms / 1000, self.audioElement.duration || 0));
                self.audioElement.currentTime = targetSec;
            }

            this.audioElement.addEventListener('loadedmetadata', () => {
                self.audioDuration = self.audioElement.duration * 1000;
                $msInput.attr('max', Math.floor(self.audioDuration));
                $secInput.attr('max', Math.floor(self.audioDuration / 1000));
                $totalTime.text(self.formatTime(self.audioDuration));

                // Restore to initial position if set and valid
                if (initialPosition > 0 && initialPosition <= self.audioDuration) {
                    updateDisplays(initialPosition);
                    seekAudioTo(initialPosition);
                } else {
                    updateDisplays(0);
                }

                self.audioLoaded = true;
            });

            this.audioElement.addEventListener('timeupdate', () => {
                // Only update display if user is not dragging the slider
                if (!self.sliderDragging && self.audioLoaded) {
                    const ms = self.audioElement.currentTime * 1000;
                    updateDisplays(ms);
                }
            });

            // Custom slider mouse handling
            // Only updates visual display - does NOT seek audio during drag
            function handleSliderInteraction(e) {
                if (!self.audioLoaded || self.audioDuration <= 0) return 0;

                const rect = $sliderTrack[0].getBoundingClientRect();
                const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                const newMs = Math.floor(percent * self.audioDuration);

                // Only update visual display - don't seek yet
                updateDisplays(newMs);

                return newMs;
            }

            // Track if audio was playing before user interaction
            let wasPlaying = false;

            $sliderTrack.on('mousedown', function (e) {
                e.preventDefault();
                e.stopPropagation();

                self.sliderDragging = true;

                // Remember if audio was playing and pause it
                wasPlaying = !self.audioElement.paused;
                if (wasPlaying) {
                    self.audioElement.pause();
                }

                // Handle the click - only updates display, no seek yet
                handleSliderInteraction(e);

                function onMouseMove(moveEvent) {
                    // Only update display while dragging - no audio seek
                    handleSliderInteraction(moveEvent);
                }

                function onMouseUp() {
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);

                    // Now do the actual seek
                    seekAudioTo(self.currentMs);

                    // If was playing, wait for seek to settle then resume
                    if (wasPlaying) {
                        // Use seeked event to know when seek is complete
                        const onSeeked = () => {
                            self.audioElement.removeEventListener('seeked', onSeeked);
                            self.audioElement.play();
                            self.sliderDragging = false;
                        };
                        self.audioElement.addEventListener('seeked', onSeeked);

                        // Fallback in case seeked doesn't fire
                        setTimeout(() => {
                            self.audioElement.removeEventListener('seeked', onSeeked);
                            if (self.audioElement.paused && wasPlaying) {
                                self.audioElement.play();
                            }
                            self.sliderDragging = false;
                        }, 300);
                    } else {
                        self.sliderDragging = false;
                    }
                }

                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            });

            // Custom volume slider handling
            function handleVolumeInteraction(e) {
                const rect = $volumeTrack[0].getBoundingClientRect();
                const percent = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));

                self.audioElement.volume = percent;
                $volumeFill.css('width', (percent * 100) + '%');
                $volumeThumb.css('left', (percent * 100) + '%');
            }

            $volumeTrack.on('mousedown', function (e) {
                e.preventDefault();

                handleVolumeInteraction(e);

                function onMouseMove(moveEvent) {
                    handleVolumeInteraction(moveEvent);
                }

                function onMouseUp() {
                    document.removeEventListener('mousemove', onMouseMove);
                    document.removeEventListener('mouseup', onMouseUp);
                }

                document.addEventListener('mousemove', onMouseMove);
                document.addEventListener('mouseup', onMouseUp);
            });

            $msInput.on('change', () => {
                const ms = parseInt($msInput.val()) || 0;
                if (self.audioLoaded) {
                    updateDisplays(ms);
                    seekAudioTo(ms);
                }
            });

            $secInput.on('change', () => {
                const sec = parseInt($secInput.val()) || 0;
                const ms = sec * 1000;
                if (self.audioLoaded) {
                    updateDisplays(ms);
                    seekAudioTo(ms);
                }
            });

            $speed.on('change', () => {
                self.audioElement.playbackRate = parseFloat($speed.val());
            });

            $('#preview-play-btn').on('click', () => {
                if (self.audioElement.paused) {
                    // Simply seek to current position and play
                    if (self.audioLoaded) {
                        seekAudioTo(self.currentMs);
                        self.audioElement.play();
                    } else {
                        self.audioElement.play();
                    }
                } else {
                    self.audioElement.pause();
                }
            });

            // Test Preview button - plays 10 seconds from the selected position
            // then returns to the original position without affecting selection
            let testPreviewTimeout = null;
            let savedPosition = 0;
            let isTestingPreview = false;

            $('#preview-test-btn').on('click', () => {
                if (isTestingPreview) {
                    // Stop the test preview
                    clearTimeout(testPreviewTimeout);
                    self.audioElement.pause();

                    // Restore the saved position
                    seekAudioTo(savedPosition);
                    updateDisplays(savedPosition);

                    isTestingPreview = false;
                    $('#preview-test-btn').text('Test Preview');
                    return;
                }

                // Save current position
                savedPosition = self.currentMs;
                isTestingPreview = true;
                $('#preview-test-btn').text('Stop Test');

                // Seek to the saved position and play
                seekAudioTo(savedPosition);
                self.audioElement.play();

                // Stop after 10 seconds and restore position
                testPreviewTimeout = setTimeout(() => {
                    self.audioElement.pause();
                    seekAudioTo(savedPosition);
                    updateDisplays(savedPosition);

                    isTestingPreview = false;
                    $('#preview-test-btn').text('Test Preview');
                }, 10000);
            });

            $('#preview-set-btn').on('click', () => {
                const ms = parseInt($msInput.val()) || 0;
                $('#preview_time').val(ms);
                self.updatePreviewDisplay();
                self.closePreviewModal();
                Utils.showFlashMessage(`Preview point set to ${self.formatTime(ms)}`, 'success');
            });

            // Load audio
            this.audioElement.src = audioUrl;
            this.audioElement.load();
        },

        formatTime(ms) {
            const totalSeconds = Math.floor(ms / 1000);
            const minutes = Math.floor(totalSeconds / 60);
            const seconds = totalSeconds % 60;
            return `${minutes}:${seconds.toString().padStart(2, '0')}`;
        },

        closePreviewModal() {
            // Save the current picker position before closing
            const currentMs = parseInt($('#preview-ms-input').val()) || 0;
            if (currentMs > 0) {
                this.lastPickerPosition = currentMs;
            }

            if (this.audioElement) {
                this.audioElement.pause();
                this.audioElement = null;
            }
            $('#preview-picker-modal').remove();
        }
    };

    // Descriptor Manager
    const DescriptorManager = {
        init() {
            this.attachDropdownHandler();
            this.attachDescriptorClickHandlers();
        },

        attachDropdownHandler() {
            $('.custom-dropdown-descriptors .dropdown-header').on('click', function () {
                const $dropdown = $(this).parent();
                const dropdownContent = document.querySelector('.dropdown-content');
                $dropdown.toggleClass('open');
                if ($dropdown.hasClass('open')) {
                    Utils.smoothScroll('.custom-dropdown-descriptors');
                    dropdownContent.removeAttribute('inert');
                } else {
                    dropdownContent.setAttribute('inert', '');
                }
            });
        },

        attachDescriptorClickHandlers() {
            $('.descriptors-container').on('click', 'input[name="descriptors"]', function (e) {
                e.preventDefault();
                const $checkbox = $(this);

                if (!$checkbox.prop('disabled')) {
                    if ($checkbox.hasClass('positive-check')) {
                        $checkbox.removeClass('positive-check').addClass('negative-check');
                    } else if ($checkbox.hasClass('negative-check')) {
                        $checkbox.removeClass('negative-check');
                        $checkbox.prop('checked', false);
                        return;
                    } else {
                        $checkbox.addClass('positive-check');
                    }
                    $checkbox.prop('checked', true);
                }
            });
        }
    };

    // Configuration Manager
    const ConfigManager = {
        init() {
            $('#export-config-btn').click(() => this.exportConfiguration());
            $('#import-config-btn').click(() => $('#import-config-input').click());
            $('#reset-config-btn').click(() => this.resetToDefaults());
            $('#import-config-input').change((e) => this.handleFileImport(e));

            // Handle auto-detect checkbox toggle
            $('#auto_detect_metadata').on('change', function () {
                if ($(this).prop('checked')) {
                    // Trigger detection if audio path exists
                    SongDetection.forceDetect();
                } else {
                    // Hide detection status when disabled
                    $('#song-detection-status').hide();
                }
            });
        },

        exportConfiguration() {
            const config = this.buildConfigObject();

            if (window.pywebview?.api?.save_file) {
                this.exportToFile(config);
            } else {
                this.fallbackDownload(config);
            }
        },

        buildConfigObject() {
            const config = {
                version: "1.1",
                timestamp: new Date().toISOString(),
                settings: {},
                descriptors: { positive: [], negative: [] },
                inContextOptions: [],
                mapperList: [],
                songMetadata: {
                    artist: $('#detected_artist').val() || '',
                    title: $('#detected_title').val() || ''
                },
                beatmapCustomization: {
                    previewTime: $('#preview_time').val() || '',
                    backgroundPath: $('#background_path').val() || ''
                }
            };

            // Export form fields
            $('#inferenceForm').find('input, select, textarea').each(function () {
                const $field = $(this);
                const name = $field.attr('name');
                const type = $field.attr('type');

                if (name && type !== 'file') {
                    config.settings[name] = type === 'checkbox' ? $field.prop('checked') : $field.val();
                }
            });

            // Export descriptors
            $('input[name="descriptors"]').each(function () {
                const $checkbox = $(this);
                const value = $checkbox.val();
                if ($checkbox.hasClass('positive-check')) {
                    config.descriptors.positive.push(value);
                } else if ($checkbox.hasClass('negative-check')) {
                    config.descriptors.negative.push(value);
                }
            });

            // Export in-context options
            $('input[name="in_context_options"]:checked').each(function () {
                config.inContextOptions.push($(this).val());
            });

            // Export mapper list
            if (typeof MapperManager !== 'undefined' && MapperManager.getAll) {
                config.mapperList = MapperManager.getAll();
            }

            return config;
        },

        async exportToFile(config) {
            try {
                const filename = `mapperatorinator-config-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;

                const filePath = await window.pywebview.api.save_file(filename);
                if (!filePath) {
                    this.showConfigStatus("Export cancelled by user", "error");
                    return;
                }

                $.ajax({
                    url: "/save_config",
                    method: "POST",
                    data: {
                        file_path: filePath,
                        config_data: JSON.stringify(config, null, 2)
                    },
                    success: (response) => {
                        if (response.success) {
                            this.showConfigStatus(`Configuration exported successfully to: ${response.file_path}`, "success");
                        } else {
                            this.showConfigStatus(`Error saving config: ${response.error}`, "error");
                        }
                    },
                    error: () => {
                        this.showConfigStatus("Failed to save config to server. Using browser download instead.", "error");
                        this.fallbackDownload(config);
                    }
                });
            } catch (error) {
                console.error("Error selecting folder:", error);
                this.fallbackDownload(config);
            }
        },

        fallbackDownload(config) {
            const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `mapperatorinator-config-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            this.showConfigStatus("Configuration exported successfully (browser download)", "success");
        },

        resetToDefaults() {
            Utils.showConfirmDialog(
                "Reset All Settings",
                "Are you sure you want to reset all settings to default values? This will also clear the queue and mapper list. This cannot be undone.",
                () => {
                    // On confirm
                    Utils.resetFormToDefaults();
                    $("#model, #gamemode, #beatmap_path").trigger('change');
                    $('#audio_path, #output_path, #beatmap_path, #lora_path').trigger('blur');

                    // Clear queue
                    if (typeof QueueManager !== 'undefined') {
                        QueueManager.clear();
                    }

                    // Clear mapper list
                    if (typeof MapperManager !== 'undefined') {
                        MapperManager.clearAll();
                    }

                    // Clear song detection status
                    $('#song-detection-status').hide().text('').removeClass('success error detecting');
                    $('#detected_artist, #detected_title').val('');

                    // Clear beatmap customization (preview time and background)
                    if (typeof BeatmapCustomization !== 'undefined') {
                        BeatmapCustomization.clearCustomizations();
                    }

                    // Update UI
                    if (typeof QueueUI !== 'undefined') {
                        QueueUI.updateUI();
                    }

                    this.showConfigStatus("All settings reset to default values", "success");
                }
            );
        },

        handleFileImport(e) {
            const file = e.target.files[0];
            if (!file) return;

            if (file.type !== 'application/json' && !file.name.endsWith('.json')) {
                this.showConfigStatus("Please select a valid JSON configuration file.", "error");
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => this.importConfiguration(e.target.result);
            reader.readAsText(file);
            $(e.target).val(''); // Reset input
        },

        importConfiguration(content) {
            try {
                const config = JSON.parse(content);
                if (!config.version) {
                    throw new Error("Invalid configuration file format");
                }

                // Import settings
                if (config.settings) {
                    Object.entries(config.settings).forEach(([name, value]) => {
                        const $field = $(`[name="${name}"]`);
                        if ($field.length) {
                            if ($field.attr('type') === 'checkbox') {
                                $field.prop('checked', value);
                            } else {
                                $field.val(value);
                            }
                        }
                    });
                }

                // Import descriptors
                $('input[name="descriptors"]').removeClass('positive-check negative-check').prop('checked', false);
                if (config.descriptors) {
                    config.descriptors.positive?.forEach(value => {
                        $(`input[name="descriptors"][value="${value}"]`)
                            .addClass('positive-check').prop('checked', true);
                    });
                    config.descriptors.negative?.forEach(value => {
                        $(`input[name="descriptors"][value="${value}"]`)
                            .addClass('negative-check').prop('checked', true);
                    });
                }

                // Import in-context options
                $('input[name="in_context_options"]').prop('checked', false);
                config.inContextOptions?.forEach(value => {
                    $(`input[name="in_context_options"][value="${value}"]`).prop('checked', true);
                });

                // Import mapper list
                if (config.mapperList && typeof MapperManager !== 'undefined' && MapperManager.loadFromArray) {
                    MapperManager.loadFromArray(config.mapperList);
                }

                // Import saved song metadata (artist/title) if present
                if (config.songMetadata) {
                    if (config.songMetadata.artist) {
                        $('#detected_artist').val(config.songMetadata.artist);
                    }
                    if (config.songMetadata.title) {
                        $('#detected_title').val(config.songMetadata.title);
                    }
                    // Don't auto-detect if we already have saved metadata
                    if (config.songMetadata.artist || config.songMetadata.title) {
                        SongDetection.lastAudioPath = $('#audio_path').val().trim(); // Prevent re-detection
                    }
                }

                // Import beatmap customization (preview time and background)
                if (config.beatmapCustomization) {
                    if (config.beatmapCustomization.previewTime) {
                        $('#preview_time').val(config.beatmapCustomization.previewTime);
                        BeatmapCustomization.updatePreviewDisplay();
                    }
                    if (config.beatmapCustomization.backgroundPath) {
                        $('#background_path').val(config.beatmapCustomization.backgroundPath);
                        PathManager.updateClearButtonVisibility($('#background_path')[0]);
                        BeatmapCustomization.updateBackgroundPreview();
                    }
                }

                // Trigger updates
                $("#model, #gamemode").trigger('change');
                $('#audio_path, #output_path, #beatmap_path, #lora_path').trigger('blur');
                $('#audio_path, #output_path, #beatmap_path, #lora_path').trigger('input');

                // Only trigger song detection if checkbox is checked AND no saved metadata AND audio path exists
                if ($('#auto_detect_metadata').prop('checked') &&
                    $('#audio_path').val().trim() &&
                    !config.songMetadata?.artist &&
                    !config.songMetadata?.title) {
                    SongDetection.forceDetect();
                }

                // Update queue UI
                if (typeof QueueUI !== 'undefined') {
                    QueueUI.updateUI();
                }

                this.showConfigStatus(`Configuration imported successfully! (${config.timestamp || 'Unknown date'})`, "success");

            } catch (error) {
                console.error("Error importing configuration:", error);
                this.showConfigStatus(`Error importing configuration: ${error.message}`, "error");
            }
        },

        showConfigStatus(message, type) {
            const $status = $("#config-status");
            $status.text(message)
                .css('color', type === 'success' ? '#28a745' : '#dc3545')
                .fadeIn();
            setTimeout(() => $status.fadeOut(), 5000);
        }
    };

    // Inference Manager
    const InferenceManager = {
        isRunning: false,

        init() {
            $('#inferenceForm').submit((e) => this.handleSubmit(e));
            $('#cancel-button').click(() => this.handleCancelClick());
        },

        handleCancelClick() {
            // If queue is running, cancel the whole queue; otherwise cancel single inference
            if (QueueUI.queueRunning) {
                QueueUI.cancelQueue();
            } else {
                this.cancelInference();
            }
        },

        async handleSubmit(e) {
            e.preventDefault();

            // Prevent double-clicks and clicks during song detection
            if (this.isRunning) {
                Utils.showFlashMessage('Inference is already running. Please wait.', 'error');
                return;
            }

            if (SongDetection.isInProgress()) {
                Utils.showFlashMessage('Please wait for song detection to complete.', 'error');
                return;
            }

            // Apply placeholder values before validation
            if (!await this.validateForm()) return;

            this.resetProgress();
            this.startInference();
        },

        async validateForm() {
            PathManager.applyPlaceholderValues();

            const audioPath = $('#audio_path').val().trim();
            const beatmapPath = $('#beatmap_path').val().trim();
            const outputPath = $('#output_path').val().trim();

            if (!audioPath && !beatmapPath) {
                Utils.smoothScroll(0);
                Utils.showFlashMessage("Either 'Beatmap Path' or 'Audio Path' are required for running inference", 'error');
                return false;
            }

            if (!outputPath && !beatmapPath) {
                Utils.smoothScroll(0);
                Utils.showFlashMessage("Either 'Output Path' or 'Beatmap Path' are required for running inference", 'error');
                return false;
            }

            // Validate beatmap file type if beatmap path is provided
            if (beatmapPath && !beatmapPath.toLowerCase().endsWith('.osu')) {
                Utils.smoothScroll('#beatmap_path');
                Utils.showFlashMessage("Beatmap file must have .osu extension", 'error');
                PathManager.showInlineError('#beatmap_path', 'Must be .osu file');
                return false;
            }

            const pathsAreValid = await PathManager.validateAndAutofillPaths(true);
            if (!pathsAreValid) {
                Utils.smoothScroll(0);
                return false;
            }

            return true;
        },

        resetProgress() {
            $('#flash-container').empty();
            $("#progress_output").show();
            Utils.smoothScroll('#progress_output');

            $("#progressBarContainer, #progressTitle").show();
            $("#progressBar").css("width", "0%").removeClass('cancelled error');
            $("#beatmapLink, #errorLogLink").hide();
            $("#init_message").text("Initializing process... This may take a moment.").show();
            $("#progressTitle").text("").css('color', '');
            $("#cancel-button").hide().prop('disabled', false).text('Cancel');
            $("button[type='submit']").prop("disabled", true);

            AppState.inferenceErrorOccurred = false;
            AppState.accumulatedErrorMessages = [];
            AppState.isCancelled = false;

            if (AppState.evtSource) {
                AppState.evtSource.close();
                AppState.evtSource = null;
            }
        },

        buildFormData() {
            const formData = new FormData($("#inferenceForm")[0]);

            // Handle descriptors
            formData.delete('descriptors');
            const positiveDescriptors = [];
            const negativeDescriptors = [];

            $('input[name="descriptors"]').each(function () {
                const $cb = $(this);
                if ($cb.hasClass('positive-check')) {
                    positiveDescriptors.push($cb.val());
                } else if ($cb.hasClass('negative-check')) {
                    negativeDescriptors.push($cb.val());
                }
            });

            positiveDescriptors.forEach(val => formData.append('descriptors', val));
            negativeDescriptors.forEach(val => formData.append('negative_descriptors', val));

            // Ensure hitsounded is true for V30
            if ($("#model").val() === "v30" && !$("#option-item-hitsounded").is(':visible')) {
                formData.set('hitsounded', 'true');
            }

            return formData;
        },

        startInference() {
            this.isRunning = true;
            $("button[type='submit']").prop("disabled", true);

            $.ajax({
                url: "/start_inference",
                method: "POST",
                data: this.buildFormData(),
                processData: false,
                contentType: false,
                success: () => {
                    $("#cancel-button").show();
                    this.connectToSSE();
                },
                error: (jqXHR, textStatus, errorThrown) => {
                    console.error("Failed to start inference:", textStatus, errorThrown);
                    let errorMsg = "Failed to start inference process. Check backend console.";
                    if (jqXHR.responseJSON && jqXHR.responseJSON.message) {
                        errorMsg = jqXHR.responseJSON.message;
                    } else if (jqXHR.responseText) {
                        try {
                            const parsed = JSON.parse(jqXHR.responseText);
                            if (parsed && parsed.message) errorMsg = parsed.message;
                        } catch (e) { /* ignore parsing error */ }
                    }
                    Utils.showFlashMessage(errorMsg, 'error');
                    this.isRunning = false;
                    $("button[type='submit']").prop("disabled", false);
                    $("#cancel-button").hide();
                    $("#progress_output").hide();
                }
            });
        },

        connectToSSE() {
            console.log("Connecting to SSE stream...");
            AppState.evtSource = new EventSource("/stream_output");
            AppState.errorLogFilePath = null;

            AppState.evtSource.onmessage = (e) => this.handleSSEMessage(e);
            AppState.evtSource.onerror = (err) => this.handleSSEError(err);
            AppState.evtSource.addEventListener("error_log", (e) => {
                AppState.errorLogFilePath = e.data;
            });
            AppState.evtSource.addEventListener("end", (e) => this.handleSSEEnd(e));
            AppState.evtSource.addEventListener("file_ready", (e) => {
                // The generated file is now ready and available at this path
                const filePath = e.data;
                console.log('[Inference] File ready:', filePath);
                // Store for later reference (e.g., opening folder)
                AppState.lastGeneratedFilePath = filePath;
            });
            AppState.evtSource.addEventListener("renamed", (e) => {
                // File was renamed to a proper osu! naming convention
                const newPath = e.data;
                console.log('[Inference] File renamed to:', newPath);
                // Update internal state but don't show redundant UI message
                AppState.lastGeneratedFilePath = newPath;
            });
        },

        handleSSEMessage(e) {
            if ($("#init_message").is(":visible")) $("#init_message").hide();
            if (AppState.isCancelled) return;

            const messageData = e.data;
            const errorIndicators = [
                "Traceback (most recent call last):", "Error executing job with overrides:",
                "FileNotFoundError:", "Exception:", "Set the environment variable HYDRA_FULL_ERROR=1"
            ];

            const isErrorMessage = errorIndicators.some(indicator => messageData.includes(indicator));

            if (isErrorMessage && !AppState.inferenceErrorOccurred) {
                AppState.inferenceErrorOccurred = true;
                AppState.accumulatedErrorMessages.push(messageData);
                $("#progressTitle").text("Error Detected").css('color', 'var(--accent-color)');
                $("#progressBar").addClass('error');
            } else if (AppState.inferenceErrorOccurred) {
                AppState.accumulatedErrorMessages.push(messageData);
            } else {
                this.updateProgress(messageData);
            }
        },

        updateProgress(messageData) {
            // Update progress title based on message content
            const lowerCaseMessage = messageData.toLowerCase();
            const progressTitles = {
                "generating timing": "Generating Timing",
                "generating kiai": "Generating Kiai",
                "generating map": "Generating Map",
                "seq len": "Refining Positions"
            };

            Object.entries(progressTitles).forEach(([keyword, title]) => {
                if (lowerCaseMessage.includes(keyword)) {
                    $("#progressTitle").text(title);
                }
            });

            // Update progress bar
            const progressMatch = messageData.match(/^\s*(\d+)%\|/);
            if (progressMatch) {
                const currentPercent = parseInt(progressMatch[1].trim(), 10);
                if (!isNaN(currentPercent)) {
                    $("#progressBar").css("width", currentPercent + "%");
                }
            }

            // Check for completion message
            if (messageData.includes("Generated beatmap saved to")) {
                const parts = messageData.split("Generated beatmap saved to");
                if (parts.length > 1) {
                    const fullPath = parts[1].trim().replace(/\\/g, "/");
                    const folderPath = fullPath.substring(0, fullPath.lastIndexOf("/"));

                    $("#beatmapLinkAnchor")
                        .attr("href", "#")
                        .text("Click here to open the folder containing your map.")
                        .off("click")
                        .on("click", (e) => {
                            e.preventDefault();
                            $.get("/open_folder", { folder: folderPath })
                                .done(response => console.log("Open folder response:", response))
                                .fail(() => alert("Failed to open folder via backend."));
                        });
                    $("#beatmapLink").show();
                }
            }
        },

        handleSSEError(err) {
            console.error("EventSource failed:", err);
            if (AppState.evtSource) {
                AppState.evtSource.close();
                AppState.evtSource = null;
            }

            if (!AppState.isCancelled && !AppState.inferenceErrorOccurred) {
                AppState.inferenceErrorOccurred = true;
                AppState.accumulatedErrorMessages.push("Error: Connection to process stream lost.");
                $("#progressTitle").text("Connection Error").css('color', 'var(--accent-color)');
                $("#progressBar").addClass('error');
                Utils.showFlashMessage("Error: Connection to process stream lost.", "error");
            }

            if (!AppState.isCancelled) {
                this.isRunning = false;
                $("button[type='submit']").prop("disabled", false);
            }
            $("#cancel-button").hide();
        },

        handleSSEEnd(e) {
            console.log("Received end event from server.", e.data);
            if (AppState.evtSource) {
                AppState.evtSource.close();
                AppState.evtSource = null;
            }

            if (AppState.isCancelled) {
                $("#progressTitle, #progressBarContainer, #beatmapLink, #errorLogLink").hide();
                $("#progress_output").hide();
            } else if (AppState.inferenceErrorOccurred) {
                this.handleInferenceError();
            } else {
                $("#progressTitle").show().text("Processing Complete").css('color', '');
                $("#progressBarContainer").show();
                $("#progressBar").css("width", "100%").removeClass('error');
            }

            this.isRunning = false;
            $("button[type='submit']").prop("disabled", false);
            $("#cancel-button").hide();
            AppState.isCancelled = false;
        },

        handleInferenceError() {
            const fullErrorText = AppState.accumulatedErrorMessages.join("\\n");
            let specificError = "An error occurred during processing. Check console/logs.";

            if (fullErrorText.includes("FileNotFoundError:")) {
                const fileNotFoundMatch = fullErrorText.match(/FileNotFoundError:.*? file (.*?) not found/);
                specificError = fileNotFoundMatch?.[1] ?
                    `Error: File not found - ${fileNotFoundMatch[1].replace(/\\\\/g, '\\\\')}` :
                    "Error: A required file was not found.";
            } else if (fullErrorText.includes("HYDRA_FULL_ERROR=1")) {
                specificError = "There was an error while creating the beatmap. Check console/logs for details.";
            } else if (fullErrorText.includes("Error executing job")) {
                specificError = "There was an error starting or executing the generation task.";
            } else if (fullErrorText.includes("Connection to process stream lost")) {
                specificError = "Error: Connection to the generation process was lost.";
            }

            Utils.showFlashMessage(specificError, "error");
            $("#progressTitle").text("Processing Failed").css('color', 'var(--accent-color)').show();
            $("#progressBar").css("width", "100%").addClass('error');
            $("#progressBarContainer").show();
            $("#beatmapLink").hide();

            if (AppState.errorLogFilePath) {
                $("#errorLogLinkAnchor").off("click").on("click", (e) => {
                    e.preventDefault();
                    $.get("/open_log_file", { path: AppState.errorLogFilePath })
                        .done(response => console.log("Open log response:", response))
                        .fail(() => alert("Failed to open log file via backend."));
                });
                $("#errorLogLink").show();
            }
        },

        cancelInference() {
            const $cancelBtn = $("#cancel-button");
            $cancelBtn.prop('disabled', true).text('Cancelling...');

            $.ajax({
                url: "/cancel_inference",
                method: "POST",
                success: (response) => { // Expecting JSON response
                    AppState.isCancelled = true;
                    Utils.showFlashMessage(response.message || "Inference cancelled successfully.", "cancel-success");
                },
                error: (jqXHR) => {
                    const errorMsg = jqXHR.responseJSON?.message || "Failed to send cancel request. Unknown error.";
                    Utils.showFlashMessage(errorMsg, "error");
                    $cancelBtn.prop('disabled', false).text('Cancel');
                }
            });
        }
    };

    // ══════════════════════════════════════════════════════════════════════════
    // QUEUE SYSTEM INTEGRATION
    // ══════════════════════════════════════════════════════════════════════════

    // Mapper Lookup Manager
    const MapperLookup = {
        cache: {},
        pendingLookups: {},

        init() {
            $('#mapper_id').on('blur', () => this.lookupCurrentMapper());
            $('#mapper_id').on('input', () => this.clearDisplay());
        },

        clearDisplay() {
            $('#mapper_name_display').removeClass('visible loading error').text('');
            $('#mapper_name').val('');
        },

        async lookupCurrentMapper() {
            const mapperId = $('#mapper_id').val().trim();
            console.log('[MapperLookup] Looking up mapper:', mapperId);
            if (!mapperId) {
                this.clearDisplay();
                return;
            }

            const $display = $('#mapper_name_display');
            console.log('[MapperLookup] Display element found:', $display.length > 0);

            // Check cache first
            if (this.cache[mapperId]) {
                console.log('[MapperLookup] Using cached value:', this.cache[mapperId]);
                $display.text(`(${this.cache[mapperId]})`).removeClass('loading error').addClass('visible');
                $('#mapper_name').val(this.cache[mapperId]);
                return;
            }

            // Show loading state
            $display.text('Looking up...').removeClass('error visible').addClass('loading visible');

            try {
                const name = await this.lookup(mapperId);
                console.log('[MapperLookup] Lookup returned:', name);
                if (name) {
                    this.cache[mapperId] = name;
                    $display.text(`(${name})`).removeClass('loading error').addClass('visible');
                    $('#mapper_name').val(name);
                    console.log('[MapperLookup] Display updated to:', $display.text());
                } else {
                    $display.text('(Not found)').removeClass('loading visible').addClass('error visible');
                    $('#mapper_name').val('');
                }
            } catch (error) {
                console.error('Mapper lookup error:', error);
                $display.text('(Lookup failed)').removeClass('loading visible').addClass('error visible');
                $('#mapper_name').val('');
            }
        },

        async lookup(mapperId) {
            // Avoid duplicate requests
            if (this.pendingLookups[mapperId]) {
                return this.pendingLookups[mapperId];
            }

            this.pendingLookups[mapperId] = new Promise((resolve, reject) => {
                $.ajax({
                    url: '/lookup_mapper_name',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ mapper_id: mapperId }),
                    success: (response) => {
                        console.log('[MapperLookup] Response:', response);
                        delete this.pendingLookups[mapperId];
                        resolve(response.username || null);
                    },
                    error: (xhr) => {
                        console.error('[MapperLookup] Error:', xhr.responseJSON);
                        delete this.pendingLookups[mapperId];
                        reject(new Error(xhr.responseJSON?.error || 'Lookup failed'));
                    }
                });
            });

            return this.pendingLookups[mapperId];
        }
    };

    // Difficulty Name Generator
    const DifficultyNameGenerator = {
        init() {
            // Listen for star rating changes
            $('#difficulty').on('input change', () => this.updateDifficultyName());

            // Listen for checkbox changes
            $('#auto_generate_diff_name').on('change', () => {
                if ($('#auto_generate_diff_name').prop('checked')) {
                    this.updateDifficultyName();
                }
            });
        },

        getDifficultyName(stars) {
            stars = parseFloat(stars) || 5.0;
            if (stars < 2.0) return 'Easy';
            if (stars < 2.7) return 'Normal';
            if (stars < 4.0) return 'Hard';
            if (stars < 5.3) return 'Insane';
            if (stars < 6.5) return 'Expert';
            return 'Expert+';
        },

        updateDifficultyName() {
            if (!$('#auto_generate_diff_name').prop('checked')) {
                return;
            }

            const stars = $('#difficulty').val();
            const diffName = this.getDifficultyName(stars);
            $('#diff_name').val(diffName);
        }
    };

    // Song Detection Manager
    const SongDetection = {
        lastAudioPath: '',
        isDetecting: false,

        init() {
            // Listen for audio path changes to trigger detection (only if checkbox is checked)
            $('#audio_path').on('blur', () => this.detectSongInfo());
        },

        async detectSongInfo() {
            // Only detect if the auto-detect checkbox is checked
            if (!$('#auto_detect_metadata').prop('checked')) {
                return;
            }

            const audioPath = $('#audio_path').val().trim();
            if (!audioPath || audioPath === this.lastAudioPath) {
                return;
            }
            this.lastAudioPath = audioPath;

            const $status = $('#song-detection-status');
            const $addBtn = $('#add-to-queue-btn');
            const $runBtn = $('button[type="submit"]');

            // Mark as detecting and disable buttons
            this.isDetecting = true;
            $status.html('Detecting song info... <span class="detection-cancel" title="Cancel detection">×</span>').removeClass('success error').addClass('detecting').show();

            // Add click handler for cancel button
            $status.find('.detection-cancel').on('click', (e) => {
                e.stopPropagation();
                this.cancelDetection();
            });
            $addBtn.prop('disabled', true).addClass('detecting');
            $runBtn.prop('disabled', true);

            // Update queue UI buttons to disable mapper buttons during detection
            if (typeof QueueUI !== 'undefined') {
                QueueUI.updateButtons();
            }

            // Clear previous values before detection
            $('#detected_artist').val('');
            $('#detected_title').val('');

            // Reset cancelled flag
            this.detectionCancelled = false;

            try {
                // Use validate_paths which now includes song detection
                const response = await $.ajax({
                    url: '/validate_paths',
                    method: 'POST',
                    data: { audio_path: audioPath, beatmap_path: '', output_path: '', detect_song: true }
                });

                // Check if cancelled during request
                if (this.detectionCancelled) {
                    return;
                }

                if (response.detected_artist) {
                    $('#detected_artist').val(response.detected_artist);
                }
                if (response.detected_title) {
                    $('#detected_title').val(response.detected_title);
                }

                if (response.detected_artist || response.detected_title) {
                    $status.text('✓ Song detected!').addClass('success');
                } else {
                    $status.text('Could not detect song info automatically.').addClass('error');
                }
            } catch (error) {
                // Check if cancelled during request
                if (this.detectionCancelled) {
                    return;
                }
                console.error('Song detection error:', error);
                $status.text('Detection failed.').addClass('error');
            } finally {
                // Only update state if not cancelled (cancelDetection already did that)
                if (!this.detectionCancelled) {
                    // Re-enable buttons
                    this.isDetecting = false;
                    $status.removeClass('detecting');
                    $('#add-to-queue-btn').removeClass('detecting');
                    $('button[type="submit"]').prop('disabled', false);
                    // Update queue UI buttons (this will re-enable Add to Queue)
                    if (typeof QueueUI !== 'undefined') {
                        QueueUI.updateButtons();
                    }
                }
            }
        },

        // Force detection (used when checkbox is toggled on)
        forceDetect() {
            this.lastAudioPath = '';
            this.detectSongInfo();
        },

        // Cancel detection in progress
        cancelDetection() {
            if (!this.isDetecting) return;

            // Mark as cancelled
            this.isDetecting = false;
            this.detectionCancelled = true;

            // Uncheck the Auto-detect Metadata checkbox
            $('#auto_detect_metadata').prop('checked', false);

            // Update status message
            const $status = $('#song-detection-status');
            $status.text('Detection cancelled').removeClass('detecting').addClass('error');

            // Re-enable buttons
            $('#add-to-queue-btn').removeClass('detecting').prop('disabled', false);
            $('button[type="submit"]').prop('disabled', false);

            // IMPORTANT: Update queue UI buttons immediately to disable mapper buttons
            if (typeof QueueUI !== 'undefined') {
                QueueUI.updateButtons();
            }
        },

        isInProgress() {
            return this.isDetecting;
        }
    };

    // Queue UI Manager
    const QueueUI = {
        queueRunning: false,

        init() {
            // Ensure QueueManager exists (from queue_manager.js)
            if (typeof QueueManager === 'undefined') {
                console.warn('QueueManager not loaded. Queue features disabled.');
                $('#queue_panel, #mapper_panel').hide();
                return;
            }

            this.attachEventHandlers();
            this.updateUI();
        },

        attachEventHandlers() {
            // Add to Queue button
            $('#add-to-queue-btn').on('click', () => this.addCurrentFormToQueue());

            // Run Queue button
            $('#run-queue-btn').on('click', () => this.runQueue());

            // Clear Queue button
            $('#clear-queue-btn').on('click', () => this.clearQueue());

            // Cancel Queue button
            $('#cancel-queue-button').on('click', () => this.cancelQueue());

            // Mapper panel handlers - use .off().on() to prevent duplicate bindings
            $('#add-mapper-btn').off('click').on('click', () => this.addMapper());
            $('#generate-from-mappers-btn').off('click').on('click', () => this.generateTasksFromMappers());

            // Listen for queue state changes from QueueManager
            window.addEventListener('queueStateChanged', () => this.updateUI());

            // Listen for audio path changes to update Add to Queue button state
            $('#audio_path').off('input blur change').on('input blur change', () => this.updateButtons());

            // Enter key for adding mapper - use .off().on() to prevent duplicate bindings
            $('#add-mapper-id').off('keypress').on('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    this.addMapper();
                }
            });
        },

        addCurrentFormToQueue() {
            // Check if song detection is still in progress
            if (SongDetection.isInProgress()) {
                Utils.showFlashMessage('Please wait for song detection to complete.', 'error');
                return;
            }

            // Collect current form data
            const formData = {};
            $('#inferenceForm').find('input, select, textarea').each(function () {
                const $field = $(this);
                const name = $field.attr('name');
                const type = $field.attr('type');

                if (name && type !== 'file') {
                    formData[name] = type === 'checkbox' ? $field.prop('checked') : $field.val();
                }
            });

            // Add descriptors
            formData.descriptors = { positive: [], negative: [] };
            $('input[name="descriptors"]').each(function () {
                const $cb = $(this);
                if ($cb.hasClass('positive-check')) {
                    formData.descriptors.positive.push($cb.val());
                } else if ($cb.hasClass('negative-check')) {
                    formData.descriptors.negative.push($cb.val());
                }
            });

            // Get mapper info
            const mapperName = $('#mapper_name').val() || '';
            const mapperId = $('#mapper_id').val() || '';
            formData.mapper_name = mapperName;
            formData.mapper_id = mapperId;

            // Build display name for queue item (use detected_artist/detected_title from form)
            const artist = formData.detected_artist || formData.song_artist || formData.artist || '??';
            const title = formData.detected_title || formData.song_title || formData.title || '??';
            const model = formData.model || 'v30';
            const creator = mapperName || `Mapperatorinator ${model.toUpperCase()}`;
            const stars = parseFloat(formData.difficulty) || 5.0;
            const baseDiff = stars < 2.0 ? 'Easy' : stars < 2.7 ? 'Normal' : stars < 4.0 ? 'Hard' : stars < 5.3 ? 'Insane' : stars < 6.5 ? 'Expert' : 'Expert+';
            const diffName = formData.diff_name || formData.difficulty_name || (mapperId ? `${creator}'s ${baseDiff}` : baseDiff);

            // Check for duplicates and append number if needed
            const baseDisplayName = `${artist} - ${title} (${creator}) [${diffName}]`;
            const existingTasks = QueueManager.getAllTasks();
            let count = 0;
            existingTasks.forEach(task => {
                const existingName = task.formData.display_name || '';
                // Match base name or numbered variants like "name (1)", "name (2)"
                if (existingName === baseDisplayName || existingName.match(new RegExp(`^${baseDisplayName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} \\(\\d+\\)$`))) {
                    count++;
                }
            });

            formData.display_name = count > 0 ? `${baseDisplayName} (${count})` : baseDisplayName;

            // Add to queue
            QueueManager.addTask(formData);
            this.updateUI();
            Utils.showFlashMessage('Task added to queue!', 'success');
        },

        async runQueue() {
            if (QueueManager.isEmpty()) {
                Utils.showFlashMessage('Queue is empty!', 'error');
                return;
            }

            // Reset queue cancelled flag
            await $.ajax({ url: '/reset_queue', method: 'POST' });

            // Check if we should compile as beatmap set
            const compileAsBeatmapSet = $('#compile-as-beatmapset').prop('checked');

            // Initialize beatmap set tracking if enabled
            if (compileAsBeatmapSet) {
                await $.ajax({
                    url: '/init_beatmapset',
                    method: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ enabled: true })
                });
            }

            // Disable run button, clear button, and compile checkbox during queue execution
            $('#run-queue-btn').prop('disabled', true);
            $('#clear-queue-btn').prop('disabled', true).addClass('disabled');
            $('#compile-as-beatmapset').prop('disabled', true);
            // Add visual disabled state and tooltip to checkbox label
            $('#compile-as-beatmapset').closest('.checkbox-label')
                .addClass('disabled')
                .attr('title', 'Cannot change while queue is running');
            this.queueRunning = true;

            // Initialize overall queue progress
            this.totalTasksAtStart = QueueManager.getAllTasks().length;
            this.completedTasksForProgress = 0;

            // Show the queue progress info panel
            $('#queue_progress_info').show();
            this.updateOverallQueueProgress(0, 0); // Reset progress

            const tasks = QueueManager.getAllTasks();
            let completedCount = 0;
            let queueCancelled = false;

            for (let i = 0; i < tasks.length; i++) {
                // Check if cancelled before starting task
                const statusResponse = await $.ajax({ url: '/queue_status', method: 'GET' });
                if (statusResponse.cancelled) {
                    queueCancelled = true;
                    // Don't show message here - cancelQueue already showed it
                    break;
                }

                const task = tasks[i];
                QueueManager.setTaskStatus(task.id, 'running');
                this.currentRunningTaskId = task.id;
                this.updateUI();

                try {
                    await this.runSingleTask(task);

                    // Check again after task completes in case user cancelled during task
                    const postTaskStatus = await $.ajax({ url: '/queue_status', method: 'GET' });
                    if (postTaskStatus.cancelled) {
                        queueCancelled = true;
                        // Task completed but queue was cancelled - mark as pending so it can run again
                        QueueManager.setTaskStatus(task.id, 'pending');
                        // Don't show message here - cancelQueue already showed it
                        break;
                    }

                    QueueManager.setTaskStatus(task.id, 'completed');
                    completedCount++;
                    this.completedTasksForProgress++;
                    // Update overall progress (task completed, 100% of this task)
                    this.updateOverallQueueProgress(this.completedTasksForProgress, 100);
                } catch (error) {
                    console.error('Task failed:', error);

                    // Check if this was a skip request (not full queue cancel)
                    const errorStatus = await $.ajax({ url: '/queue_status', method: 'GET' });
                    if (errorStatus.cancelled) {
                        queueCancelled = true;
                        // Task was interrupted - mark as pending so it can run again
                        QueueManager.setTaskStatus(task.id, 'pending');
                        // Don't show message here - cancelQueue already showed it
                        break;
                    }

                    // Task was skipped but queue continues
                    if (error.message === 'Task skipped') {
                        QueueManager.setTaskStatus(task.id, 'skipped');
                        Utils.showFlashMessage('Task skipped.', 'cancel-success');
                    } else {
                        QueueManager.setTaskStatus(task.id, 'failed');
                    }
                    // Even failed/skipped tasks count towards progress
                    this.completedTasksForProgress++;
                    this.updateOverallQueueProgress(this.completedTasksForProgress, 100);
                }

                this.updateUI();
            }

            // Clean up
            this.queueRunning = false;
            this.currentRunningTaskId = null;

            // Re-enable controls
            $('#run-queue-btn').prop('disabled', false);
            $('#clear-queue-btn').prop('disabled', false).removeClass('disabled');
            $('#compile-as-beatmapset').prop('disabled', false);
            // Remove visual disabled state and tooltip from checkbox label
            $('#compile-as-beatmapset').closest('.checkbox-label')
                .removeClass('disabled')
                .removeAttr('title');

            // Hide queue progress info after a delay (let user see final status)
            setTimeout(() => {
                if (!this.queueRunning) {
                    $('#queue_progress_info').hide();
                }
            }, 3000);

            this.updateUI(); // Ensure UI is up to date

            if (queueCancelled) {
                // Don't clear the queue or show additional messages
                // cancelQueue() already showed the notification
                // Tasks remain so user can run again
            } else {
                // Finalize beatmap set if enabled
                if (compileAsBeatmapSet && completedCount > 0) {
                    try {
                        const response = await $.ajax({
                            url: '/finalize_beatmapset',
                            method: 'POST'
                        });
                        if (response.success) {
                            Utils.showFlashMessage(`Queue completed: ${completedCount}/${tasks.length} tasks. Beatmap set created: ${response.filename}`, 'success');
                        } else {
                            Utils.showFlashMessage(`Queue completed: ${completedCount}/${tasks.length} tasks. Beatmap set creation failed.`, 'error');
                        }
                    } catch (error) {
                        console.error('Failed to finalize beatmap set:', error);
                        Utils.showFlashMessage(`Queue completed: ${completedCount}/${tasks.length} tasks successful.`, 'success');
                    }
                } else {
                    Utils.showFlashMessage(`Queue completed: ${completedCount}/${tasks.length} tasks successful.`, 'success');
                }
            }
        },

        runSingleTask(task) {
            return new Promise((resolve, reject) => {
                this.currentTaskResolve = resolve;
                this.currentTaskReject = reject;

                // Build form data from task
                const formData = new FormData();
                Object.entries(task.formData).forEach(([key, value]) => {
                    if (key === 'descriptors') {
                        value.positive?.forEach(v => formData.append('descriptors', v));
                        value.negative?.forEach(v => formData.append('negative_descriptors', v));
                    } else if (typeof value === 'boolean') {
                        formData.append(key, value.toString());
                    } else if (value !== null && value !== undefined && value !== '') {
                        formData.append(key, value);
                    }
                });

                // Start inference
                $.ajax({
                    url: '/start_inference',
                    method: 'POST',
                    data: formData,
                    processData: false,
                    contentType: false,
                    success: () => {
                        // Connect to SSE and wait for completion
                        this.waitForTaskCompletion(resolve, reject);
                    },
                    error: (xhr) => {
                        reject(new Error(xhr.responseJSON?.message || 'Failed to start task'));
                    }
                });
            });
        },

        waitForTaskCompletion(resolve, reject) {
            const evtSource = new EventSource('/stream_output');
            let completed = false;
            this.currentEvtSource = evtSource;

            InferenceManager.resetProgress();
            $('#progress_output').show();

            // Show cancel button with appropriate text
            $('#cancel-button').show().text(this.queueRunning ? 'Cancel Queue' : 'Cancel');

            evtSource.onmessage = (e) => {
                if ($("#init_message").is(":visible")) $("#init_message").hide();
                InferenceManager.updateProgress(e.data);

                // Update overall queue progress based on current task progress
                const percentMatch = e.data.match(/(\d+(?:\.\d+)?)\s*%/);
                if (percentMatch) {
                    const percent = parseFloat(percentMatch[1]);
                    this.updateOverallQueueProgress(this.completedTasksForProgress, percent);
                }
            };

            evtSource.addEventListener('end', () => {
                evtSource.close();
                this.currentEvtSource = null;
                completed = true;
                $('#cancel-button').hide();
                $("button[type='submit']").prop("disabled", false);

                if (AppState.inferenceErrorOccurred) {
                    reject(new Error('Inference failed'));
                } else {
                    resolve();
                }
            });

            evtSource.addEventListener('renamed', (e) => {
                // File renamed - just log it, no UI display needed
                console.log('[Queue] File renamed:', e.data);
            });

            evtSource.addEventListener('file_ready', (e) => {
                // The generated file is now ready and available at this path
                const filePath = e.data;
                console.log('[Queue] File ready:', filePath);

                // Update the queue item to show the file is ready
                if (this.currentRunningTaskId) {
                    const $item = $(`.queue-item[data-task-id="${this.currentRunningTaskId}"]`);
                    $item.addClass('file-ready');

                    // Store the file path on the task for later reference
                    const tasks = QueueManager.getAllTasks();
                    const task = tasks.find(t => t.id === this.currentRunningTaskId);
                    if (task && task.formData) {
                        task.formData._generatedFilePath = filePath;
                        // Note: Beatmapset tracking is handled by stream_output() in the backend
                        // to avoid duplicate entries
                    }
                }
            });

            evtSource.onerror = () => {
                evtSource.close();
                this.currentEvtSource = null;
                if (!completed) {
                    reject(new Error('Connection lost'));
                }
            };
        },

        updateOverallQueueProgress(completedTasks, currentTaskProgress) {
            // Calculate overall queue progress
            // Each completed task = 100%, current task = currentTaskProgress%
            // Total = (completedTasks * 100 + currentTaskProgress) / (totalTasks * 100) * 100
            const totalTasks = this.totalTasksAtStart || 1;
            const overallPercent = ((completedTasks * 100) + currentTaskProgress) / (totalTasks * 100) * 100;

            // Update the #queue_task_label to show overall queue progress
            $('#queue_task_label').text(`Queue: ${completedTasks}/${totalTasks} tasks (${Math.round(overallPercent)}%)`);
        },

        cancelQueue() {
            $.ajax({
                url: '/cancel_inference',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ clear_queue: true }),
                success: (response) => {
                    // Hide progress output completely
                    $('#progress_output').hide();
                    $('#cancel-button').hide();
                    $('#queue_progress_info').hide();

                    // Reset current running task status back to pending (not cancelled/shaded)
                    if (this.currentRunningTaskId) {
                        QueueManager.setTaskStatus(this.currentRunningTaskId, 'pending');
                        this.updateUI();
                    }

                    // Re-enable checkbox and remove disabled styling
                    $('#compile-as-beatmapset').prop('disabled', false);
                    $('#compile-as-beatmapset').closest('.checkbox-label')
                        .removeClass('disabled')
                        .removeAttr('title');

                    Utils.showFlashMessage(response.message || 'Queue cancelled.', 'cancel-success');
                },
                error: (xhr) => {
                    Utils.showFlashMessage(xhr.responseJSON?.message || 'Failed to cancel queue.', 'error');
                }
            });
        },

        skipCurrentTask() {
            // Skip (cancel) only the current running task
            $.ajax({
                url: '/cancel_inference',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ clear_queue: false }),
                success: () => {
                    if (this.currentTaskReject) {
                        this.currentTaskReject(new Error('Task skipped'));
                    }
                },
                error: (xhr) => {
                    Utils.showFlashMessage(xhr.responseJSON?.message || 'Failed to skip task.', 'error');
                }
            });
        },

        clearQueue() {
            if (QueueManager.isEmpty()) return;

            QueueManager.clearAll();
            this.updateUI();

            // Don't show or reset progress output - just clear the queue
            Utils.showFlashMessage('Queue cleared.', 'success');
        },

        async addMapper() {
            const mapperId = $('#add-mapper-id').val().trim();
            $('#add-mapper-id').val('');  // Clear immediately

            if (!mapperId) {
                Utils.showFlashMessage('Please enter a mapper ID.', 'error');
                return;
            }

            // Look up mapper name
            let mapperName = MapperLookup.cache[mapperId];
            if (!mapperName) {
                try {
                    mapperName = await MapperLookup.lookup(mapperId);
                } catch (error) {
                    console.error('Mapper lookup failed:', error);
                }
            }

            MapperManager.addMapper(mapperId, mapperName || 'Unknown', 1);
            this.updateUI();
            Utils.showFlashMessage(`Mapper added: ${mapperName || mapperId}`, 'success');
        },

        // Add a single mapper to queue (from the + button in mapper list)
        addSingleMapperToQueue(mapperId, mapperName, count) {
            // Get current form data as template
            const templateData = {};
            $('#inferenceForm').find('input, select, textarea').each(function () {
                const $field = $(this);
                const name = $field.attr('name');
                const type = $field.attr('type');

                if (name && type !== 'file') {
                    templateData[name] = type === 'checkbox' ? $field.prop('checked') : $field.val();
                }
            });

            // Add descriptors
            templateData.descriptors = { positive: [], negative: [] };
            $('input[name="descriptors"]').each(function () {
                const $cb = $(this);
                if ($cb.hasClass('positive-check')) {
                    templateData.descriptors.positive.push($cb.val());
                } else if ($cb.hasClass('negative-check')) {
                    templateData.descriptors.negative.push($cb.val());
                }
            });

            // Get metadata for display name
            const artist = templateData.detected_artist || templateData.song_artist || templateData.artist || '??';
            const title = templateData.detected_title || templateData.song_title || templateData.title || '??';
            const model = templateData.model || 'v30';
            const stars = parseFloat(templateData.difficulty) || 5.0;
            const baseDiff = DifficultyNameGenerator.getDifficultyName(stars);

            // Generate tasks
            let tasksAdded = 0;
            const existingTasks = QueueManager.getAllTasks();

            for (let i = 0; i < count; i++) {
                const taskData = { ...templateData };
                taskData.mapper_id = mapperId;
                taskData.mapper_name = mapperName;

                // Compose difficulty name with mapper name
                const userDiff = templateData.diff_name?.trim();
                let diffName;
                if (userDiff) {
                    diffName = count > 1 ? `${mapperName}'s ${userDiff} ${i + 1}` : `${mapperName}'s ${userDiff}`;
                } else {
                    diffName = count > 1 ? `${mapperName}'s ${baseDiff} ${i + 1}` : `${mapperName}'s ${baseDiff}`;
                }
                taskData.diff_name = diffName;

                // Build proper display name: Artist - Title (Mapperatorinator V##) [Mapper's Difficulty]
                const creator = `Mapperatorinator ${model.toUpperCase()}`;
                const baseDisplayName = `${artist} - ${title} (${creator}) [${diffName}]`;

                // Check for duplicates
                let dupCount = 0;
                existingTasks.forEach(task => {
                    const existingName = task.formData.display_name || '';
                    if (existingName === baseDisplayName || existingName.match(new RegExp(`^${baseDisplayName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} \\(\\d+\\)$`))) {
                        dupCount++;
                    }
                });

                taskData.display_name = dupCount > 0 ? `${baseDisplayName} (${dupCount})` : baseDisplayName;

                QueueManager.addTask(taskData);
                tasksAdded++;
            }

            this.updateUI();
            Utils.showFlashMessage(`Added ${tasksAdded} task${tasksAdded !== 1 ? 's' : ''} for ${mapperName}`, 'success');
        },

        generateTasksFromMappers() {
            const mappers = MapperManager.getAll();
            if (mappers.length === 0) {
                Utils.showFlashMessage('No mappers in list!', 'error');
                return;
            }

            // Get current form data as template
            const templateData = {};
            $('#inferenceForm').find('input, select, textarea').each(function () {
                const $field = $(this);
                const name = $field.attr('name');
                const type = $field.attr('type');

                if (name && type !== 'file') {
                    templateData[name] = type === 'checkbox' ? $field.prop('checked') : $field.val();
                }
            });

            // Add descriptors
            templateData.descriptors = { positive: [], negative: [] };
            $('input[name="descriptors"]').each(function () {
                const $cb = $(this);
                if ($cb.hasClass('positive-check')) {
                    templateData.descriptors.positive.push($cb.val());
                } else if ($cb.hasClass('negative-check')) {
                    templateData.descriptors.negative.push($cb.val());
                }
            });

            // Get metadata for display name
            const artist = templateData.detected_artist || templateData.song_artist || templateData.artist || '??';
            const title = templateData.detected_title || templateData.song_title || templateData.title || '??';
            const model = templateData.model || 'v30';
            const stars = parseFloat(templateData.difficulty) || 5.0;
            const baseDiff = DifficultyNameGenerator.getDifficultyName(stars);

            // Generate tasks for each mapper
            let totalTasks = 0;
            const existingTasks = QueueManager.getAllTasks();

            mappers.forEach(mapper => {
                if (!mapper.checked) return; // Skip unchecked mappers

                const qty = mapper.n || 1;
                for (let i = 0; i < qty; i++) {
                    const taskData = { ...templateData };
                    taskData.mapper_id = mapper.id;
                    taskData.mapper_name = mapper.name;

                    // Compose difficulty name with mapper name
                    const userDiff = templateData.diff_name?.trim();
                    let diffName;
                    if (userDiff) {
                        diffName = qty > 1 ? `${mapper.name}'s ${userDiff} ${i + 1}` : `${mapper.name}'s ${userDiff}`;
                    } else {
                        diffName = qty > 1 ? `${mapper.name}'s ${baseDiff} ${i + 1}` : `${mapper.name}'s ${baseDiff}`;
                    }
                    taskData.diff_name = diffName;

                    // Build proper display name: Artist - Title (Mapperatorinator V##) [Mapper's Difficulty]
                    const creator = `Mapperatorinator ${model.toUpperCase()}`;
                    const baseDisplayName = `${artist} - ${title} (${creator}) [${diffName}]`;

                    // Check for duplicates
                    let count = 0;
                    existingTasks.forEach(task => {
                        const existingName = task.formData.display_name || '';
                        if (existingName === baseDisplayName || existingName.match(new RegExp(`^${baseDisplayName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} \\(\\d+\\)$`))) {
                            count++;
                        }
                    });

                    taskData.display_name = count > 0 ? `${baseDisplayName} (${count})` : baseDisplayName;

                    QueueManager.addTask(taskData);
                    totalTasks++;
                }
            });

            this.updateUI();
            Utils.showFlashMessage(`Generated ${totalTasks} tasks from mapper list!`, 'success');
        },

        updateUI() {
            this.updateQueueList();
            this.updateMapperList();
            this.updateButtons();
        },

        updateQueueList() {
            const $list = $('#queue-list');
            const tasks = QueueManager.getAllTasks();

            if (tasks.length === 0) {
                $list.html('<div class="queue-empty">No tasks in queue. Add tasks using "Add to Queue" button.</div>');
                $('#queue-count').text('0 tasks');
                return;
            }

            let html = '';
            tasks.forEach((task, index) => {
                const displayName = task.formData.display_name || 'Unnamed Task';
                const statusClass = task.status || '';
                // No status icons - keep it clean
                const statusIcon = task.status === 'completed' ? '✓ ' : task.status === 'skipped' ? '⏭ ' : '';
                const fd = task.formData;
                const isRunning = task.status === 'running';

                // Build categorized details
                // Row 1: Basic Settings (Model, Gamemode, Difficulty)
                const basicSettings = [];
                if (fd.model) basicSettings.push({ label: 'Model', value: fd.model.toUpperCase() });
                if (fd.gamemode !== undefined) basicSettings.push({ label: 'Mode', value: this.getGamemodeName(fd.gamemode) });
                if (fd.difficulty) basicSettings.push({ label: 'Stars', value: fd.difficulty });

                // Row 2: Difficulty Metadata (HP, CS, OD, AR, Slider Mult, Tick Rate)
                const diffMetadata = [];
                if (fd.hp_drain_rate) diffMetadata.push({ label: 'HP', value: fd.hp_drain_rate });
                if (fd.circle_size) diffMetadata.push({ label: 'CS', value: fd.circle_size });
                if (fd.overall_difficulty) diffMetadata.push({ label: 'OD', value: fd.overall_difficulty });
                if (fd.approach_rate) diffMetadata.push({ label: 'AR', value: fd.approach_rate });
                if (fd.slider_multiplier) diffMetadata.push({ label: 'SV', value: fd.slider_multiplier });
                if (fd.slider_tick_rate) diffMetadata.push({ label: 'Tick', value: fd.slider_tick_rate });

                // Row 3: Advanced Settings (Mapper ID, Year (V31+ only), CFG, Temp, Top-p, Seed)
                const advancedSettings = [];
                if (fd.mapper_id) advancedSettings.push({ label: 'Mapper ID', value: fd.mapper_id });
                // Only show Year if model supports it (V31+)
                const modelCaps = AppState.modelCapabilities[fd.model] || {};
                if (fd.year && modelCaps.supportsYear !== false) {
                    advancedSettings.push({ label: 'Year', value: fd.year });
                }
                if (fd.cfg_scale) advancedSettings.push({ label: 'CFG', value: fd.cfg_scale });
                if (fd.temperature) advancedSettings.push({ label: 'Temp', value: fd.temperature });
                if (fd.top_p) advancedSettings.push({ label: 'Top-p', value: fd.top_p });
                if (fd.seed) advancedSettings.push({ label: 'Seed', value: fd.seed });

                // Row 4: Generation Interval (Start, End)
                const genInterval = [];
                if (fd.start_time) genInterval.push({ label: 'Start', value: `${fd.start_time}ms` });
                if (fd.end_time) genInterval.push({ label: 'End', value: `${fd.end_time}ms` });

                // Row 5: Beatmap Customization (Preview Time, Background)
                const beatmapCustom = [];
                if (fd.preview_time) beatmapCustom.push({ label: 'Preview', value: `${fd.preview_time}ms` });
                if (fd.background_path) {
                    // Just show filename for display
                    const bgFilename = fd.background_path.split(/[/\\]/).pop();
                    beatmapCustom.push({ label: 'Background', value: bgFilename });
                }

                // Row 6: Descriptors (V31+ only, if any are set)
                const descriptorTags = [];
                if (fd.descriptors && modelCaps.supportsDescriptors !== false) {
                    const desc = fd.descriptors;
                    if (desc.positive && desc.positive.length > 0) {
                        desc.positive.forEach(d => descriptorTags.push({ label: '+', value: d }));
                    }
                    if (desc.negative && desc.negative.length > 0) {
                        desc.negative.forEach(d => descriptorTags.push({ label: '−', value: d }));
                    }
                }

                // Build HTML for each category row
                const buildRow = (items, label) => {
                    if (items.length === 0) return '';
                    const cells = items.map(item =>
                        `<div class="queue-detail-cell"><span class="detail-label">${item.label}:</span><span class="detail-value">${item.value}</span></div>`
                    ).join('');
                    return `<div class="queue-detail-section">
                        <span class="queue-detail-section-label">${label}</span>
                        <div class="queue-detail-row">${cells}</div>
                    </div>`;
                };

                // Build background preview HTML if background is set
                let backgroundPreviewHtml = '';
                if (fd.background_path) {
                    backgroundPreviewHtml = `
                        <div class="queue-item-bg-preview">
                            <img src="/preview_background?path=${encodeURIComponent(fd.background_path)}" alt="Background" 
                                 onerror="this.parentElement.style.display='none'">
                        </div>
                    `;
                }

                let detailsHtml = '';
                detailsHtml += buildRow(basicSettings, 'Basic Settings');
                detailsHtml += buildRow(diffMetadata, 'Difficulty Metadata');
                detailsHtml += buildRow(advancedSettings, 'Advanced Settings');
                detailsHtml += buildRow(genInterval, 'Generation Interval');
                detailsHtml += buildRow(beatmapCustom, 'Beatmap Customization');
                detailsHtml += buildRow(descriptorTags, 'Descriptors');

                // X button title and action changes based on queue running state
                const removeTitle = isRunning ? 'Skip this task' : (this.queueRunning ? 'Remove from queue' : 'Remove from queue');
                const removeAction = isRunning ? `QueueUI.skipCurrentTask()` : `QueueUI.removeTask('${task.id}')`;

                html += `
                    <div class="queue-item ${statusClass}" data-task-id="${task.id}">
                        ${backgroundPreviewHtml}
                        <div class="queue-item-header">
                            <div class="queue-item-info">
                                <span class="queue-item-name queue-item-toggle">${statusIcon}${displayName}</span>
                                <span class="queue-item-arrow">▼</span>
                            </div>
                            <div class="queue-item-actions">
                                <button class="queue-item-btn remove" onclick="${removeAction}" title="${removeTitle}">×</button>
                            </div>
                        </div>
                        <div class="queue-item-details" style="display:none;">
                            ${detailsHtml}
                        </div>
                    </div>
                `;
            });

            $list.html(html);
            $('#queue-count').text(`${tasks.length} task${tasks.length !== 1 ? 's' : ''}`);

            // Attach dropdown toggle handlers
            $list.find('.queue-item-toggle, .queue-item-arrow').off('click').on('click', function (e) {
                e.stopPropagation();
                const $item = $(this).closest('.queue-item');
                const $details = $item.find('.queue-item-details');
                const $arrow = $item.find('.queue-item-arrow');

                $details.slideToggle(150);
                $arrow.toggleClass('expanded');
            });
        },

        updateMapperList() {
            // MapperManager handles its own rendering via _entryTemplate
            // We just need to update the button states
            this.updateButtons();
        },

        updateButtons() {
            const queueEmpty = QueueManager.isEmpty();
            const mappersEmpty = MapperManager.getAll().length === 0;
            const hasCheckedMappers = MapperManager.getAll().some(m => m.checked);
            const audioPath = $('#audio_path').val().trim();
            const hasAudio = audioPath !== '';
            const isDetecting = SongDetection.isInProgress();

            $('#run-queue-btn').prop('disabled', queueEmpty || this.queueRunning);

            // Disable mapper-related buttons if no audio or detecting
            const canUseMappers = hasAudio && !isDetecting;
            const $genFromMappersBtn = $('#generate-from-mappers-btn');
            $genFromMappersBtn.prop('disabled', mappersEmpty || !hasCheckedMappers || !canUseMappers);

            // Set tooltip for Queue Tasks From Selected Mappers button
            if (!hasAudio) {
                $genFromMappersBtn.attr('title', 'Load an audio file first');
            } else if (isDetecting) {
                $genFromMappersBtn.attr('title', 'Waiting for song detection to complete...');
            } else if (mappersEmpty) {
                $genFromMappersBtn.attr('title', 'Add mappers to the list first');
            } else if (!hasCheckedMappers) {
                $genFromMappersBtn.attr('title', 'Check at least one mapper');
            } else {
                $genFromMappersBtn.attr('title', 'Generate queue tasks for all checked mappers');
            }

            // Update add-single-mapper buttons (the + buttons in mapper list)
            $('.add-single-mapper-btn').prop('disabled', !canUseMappers);
            if (!canUseMappers) {
                $('.add-single-mapper-btn').attr('title', !hasAudio ? 'Load an audio file first' : 'Waiting for song detection...');
            } else {
                $('.add-single-mapper-btn').attr('title', 'Add this mapper to queue');
            }

            // Add to Queue button - disabled if no audio or still detecting
            const $addBtn = $('#add-to-queue-btn');
            if (!hasAudio) {
                $addBtn.prop('disabled', true).removeClass('detecting').attr('title', 'Load an audio file first');
            } else if (isDetecting) {
                $addBtn.prop('disabled', true).addClass('detecting').attr('title', 'Waiting for song detection to complete...');
            } else {
                $addBtn.prop('disabled', false).removeClass('detecting').attr('title', 'Add current settings to queue');
            }
        },

        getGamemodeName(mode) {
            const modes = { '0': 'osu!', '1': 'Taiko', '2': 'Catch', '3': 'Mania' };
            return modes[mode] || 'osu!';
        },

        removeTask(taskId) {
            // Just remove task from queue (only called for non-running tasks)
            QueueManager.removeTask(taskId);
            this.updateUI();
        },

        removeMapper(mapperId) {
            MapperManager.removeMapper(mapperId);
            this.updateUI();
        },

        incrementMapperQty(mapperId) {
            MapperManager.updateQuantity(mapperId, 1);
            this.updateUI();
        },

        decrementMapperQty(mapperId) {
            MapperManager.updateQuantity(mapperId, -1);
            this.updateUI();
        }
    };

    // Make QueueUI globally accessible for onclick handlers
    window.QueueUI = QueueUI;

    // GPU Capability Manager
    const GPUCapability = {
        bf16Supported: false,
        gpuName: null,

        init() {
            this.checkBf16Support();
        },

        async checkBf16Support() {
            try {
                const response = await $.ajax({
                    url: '/check_bf16_support',
                    method: 'GET'
                });

                this.bf16Supported = response.supported;
                this.gpuName = response.gpu_name;

                if (response.supported) {
                    // Show the bf16 option
                    $('#bf16-option').show();

                    // Update tooltip with GPU name
                    $('#enable_bf16').next('label').attr('title',
                        `Enable bfloat16 precision for ~40-60% faster generation on ${response.gpu_name}. No quality loss.`);
                }
            } catch (error) {
                // Keep the option hidden on error
            }
        }
    };

    // Initialize all components
    function initializeApp() {
        // Initialize Select2
        $('.select2').select2({
            placeholder: "Select options",
            allowClear: true,
            dropdownCssClass: "select2-dropdown-dark",
            containerCssClass: "select2-container-dark"
        });

        // Ensure progress title div exists
        if (!$("#progressTitle").length) {
            $("#progress_output h3").after("<div id='progressTitle' style='font-weight:bold; padding-bottom:5px;'></div>");
        }

        // Initialize all managers
        FileBrowser.init();
        PathManager.init();
        DescriptorManager.init();
        ConfigManager.init();
        InferenceManager.init();

        // Initialize queue system components
        MapperLookup.init();
        SongDetection.init();
        DifficultyNameGenerator.init();
        BeatmapCustomization.init();
        QueueUI.init();

        // Check GPU capabilities for advanced options
        GPUCapability.init();

        // Attach event handlers
        $("#model").on('change', () => UIManager.updateModelSettings());
        $("#gamemode").on('change', () => UIManager.updateConditionalFields());

        // Initial UI updates
        UIManager.updateModelSettings();
    }

    // Start the application
    initializeApp();
});
