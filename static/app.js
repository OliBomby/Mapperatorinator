$(document).ready(function() {
    // Application state and configuration
    const AppState = {
        jobs: new Map(),
        jobCounter: 0,
        lastStartedJobId: null,
        cancelCloseCooldownMs: 3000,
        nextCancelCloseAt: 0,
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

            // Clear descriptors
            $('input[name="descriptors"], input[name="in_context_options"]')
                .removeClass('positive-check negative-check').prop('checked', false);

            ValidationManager.clearPlaceholders();
            return ValidationManager.validateAndAutofill(false);
        }
    };

    // UI Manager for conditional visibility
    const UIManager = {
        clearable_inputs: '#audio_path, #beatmap_path, #output_path, #lora_path, #background_image',

        init() {
            this.attachClearButtonHandlers();
            $(this.clearable_inputs).trigger('blur');
        },

        attachClearButtonHandlers() {
            // Listen for input events (typing)
            $(this.clearable_inputs).on('input', (e) => {
                this.updateClearButtonVisibility(e.target);
            });

            // Listen for blur events (leaving field) - immediate validation
            $(this.clearable_inputs).on('blur', (e) => {
                this.updateClearButtonVisibility(e.target);
            });

            // Handle clear button clicks
            $('.clear-input-btn').on('click', (e) => {
                const targetId = $(e.target).data('target');
                const $targetInput = $(`#${targetId}`);

                $targetInput.val('');
                this.updateClearButtonVisibility($targetInput[0]);
                return ValidationManager.validateAndAutofill(false);
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

        updateConditionalFields() {
            const selectedGamemode = $("#gamemode").val();
            const selectedModel = $("#model").val();
            const beatmapPath = $('#beatmap_path').val().trim();

            // Handle gamemode-based visibility
            $('.conditional-field[data-show-for-gamemode]').each(function() {
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
            $('.conditional-field[data-hide-for-model]').each(function() {
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
                $gamemodeSelect.find("option").each(function() {
                    $(this).prop('disabled', $(this).val() !== '0');
                });
            } else {
                $gamemodeSelect.prop('disabled', false);
                $gamemodeSelect.find("option").prop('disabled', false);
            }

            // Handle in-context options
            const supportedContext = capabilities.supportedInContextOptions ||
                                   ['NONE', 'TIMING', 'KIAI', 'MAP', 'GD', 'NO_HS'];

            $('input[name="in_context_options"]').each(function() {
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

            this.updateConditionalFields();
        }
    };

    // File Browser Manager
    const FileBrowser = {
        init() {
            this.attachBrowseHandlers();
        },

        attachBrowseHandlers() {
            $('.browse-button[data-browse-type]').click(async function() {
                const browseType = $(this).data('browse-type');
                const targetId = $(this).data('target');

                try {
                    let path;

                    if (browseType === 'folder') {
                        path = await window.pywebview.api.browse_folder();
                    } else if (browseType === 'image') {
                        path = await window.pywebview.api.browse_image();
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

    const validation_trigger_inputs = '#audio_path, #beatmap_path, #output_path';

    // Path Manager for autofill, validation and clear button support
    const ValidationManager = {
        init() {
            this.attachValidationChangeHandlers();
            $(validation_trigger_inputs).trigger('blur');
        },

        attachValidationChangeHandlers() {
            // Listen for blur events (leaving field) - immediate validation
            $(validation_trigger_inputs).on('blur', (_) => {
                return this.validateAndAutofill(false);
            });
        },

        validateAndAutofill(showFlashMessages = false) { // isFileDialog replaced by showFlashMessages
            const audioPath = $('#audio_path').val().trim();
            const beatmapPath = $('#beatmap_path').val().trim();
            const outputPath = $('#output_path').val().trim();

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
                        this.clearPlaceholders();
                        resolve(false);
                    }
                });
            });
        },

        placeholder_elements: {
            '#audio_path': 'audio_path',
            '#output_path': 'output_path',
            '#beatmap_path': 'beatmap_path',
            '#gamemode': 'gamemode',
            '#difficulty': 'difficulty',
            '#title': 'title',
            '#title_unicode': 'title_unicode',
            '#artist': 'artist',
            '#artist_unicode': 'artist_unicode',
            '#creator': 'creator',
            '#version': 'version',
            '#preview_time': 'preview_time',
            '#background_image': 'background',
            '#source': 'source',
            '#tags': 'tags',
            '#hp_drain_rate': 'hp_drain_rate',
            '#circle_size': 'circle_size',
            '#approach_rate': 'approach_rate',
            '#overall_difficulty': 'overall_difficulty',
            '#slider_multiplier': 'slider_multiplier',
            '#slider_tick_rate': 'slider_tick_rate',
            '#hold_note_ratio': 'hold_note_ratio',
            '#scroll_speed_ratio': 'scroll_speed_ratio',
            '#mapper_id': 'mapper_id',
        },

        handleValidationResponse(response, showFlashMessages = false) {
            this.clearValidationErrors();
            const autofilledArgs = response.autofilled_args;

            // Show autofilled values as placeholders
            Object.entries(this.placeholder_elements).forEach(([selector, argName]) => {
                const $input = $(selector);
                if (autofilledArgs && autofilledArgs[argName] !== undefined && autofilledArgs[argName] !== null) {
                    $input.attr('placeholder', autofilledArgs[argName]);
                } else {
                    $input.attr('placeholder', '');
                }
            });

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
            Object.keys(this.placeholder_elements).forEach(selector => {
                $(selector).attr('placeholder', '');
            });
            this.clearValidationErrors();
        },
    };

    // Descriptor Manager
    const DescriptorManager = {
        init() {
            this.attachDropdownHandler();
            this.attachDescriptorClickHandlers();
        },

        attachDropdownHandler() {
            $('.custom-dropdown-descriptors .dropdown-header').on('click', function() {
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
            $('.descriptors-container').on('click', 'input[name="descriptors"]', function(e) {
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
                version: "1.0",
                timestamp: new Date().toISOString(),
                settings: {},
                descriptors: { positive: [], negative: [] },
                inContextOptions: []
            };

            // Export form fields
            $('#inferenceForm').find('input, select, textarea').each(function() {
                const $field = $(this);
                const name = $field.attr('name');
                const type = $field.attr('type');

                if (name && type !== 'file') {
                    config.settings[name] = type === 'checkbox' ? $field.prop('checked') : $field.val();
                }
            });

            // Export descriptors
            $('input[name="descriptors"]').each(function() {
                const $checkbox = $(this);
                const value = $checkbox.val();
                if ($checkbox.hasClass('positive-check')) {
                    config.descriptors.positive.push(value);
                } else if ($checkbox.hasClass('negative-check')) {
                    config.descriptors.negative.push(value);
                }
            });

            // Export in-context options
            $('input[name="in_context_options"]:checked').each(function() {
                config.inContextOptions.push($(this).val());
            });

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
            if (confirm("Are you sure you want to reset all settings to default values? This cannot be undone.")) {
                Utils.resetFormToDefaults();
                $("#model, #gamemode, #beatmap_path").trigger('change');
                $(UIManager.clearable_inputs).trigger('blur');
                this.showConfigStatus("All settings reset to default values", "success");
            }
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

                // Trigger updates
                $("#model, #gamemode").trigger('change');
                $(UIManager.clearable_inputs).trigger('blur');
                $(UIManager.clearable_inputs).trigger('input');

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
        init() {
            $('#inferenceForm').submit((e) => this.handleSubmit(e));
        },

        async handleSubmit(e) {
            e.preventDefault();

            // Apply placeholder values before validation
            if (!await this.validateForm()) return;

            this.removeFinishedCards();
            const formData = this.buildFormData();

            if (this.shouldDelayStart()) {
                Utils.showFlashMessage("Please wait until the previous job starts generating before starting a new one.", 'error');
                return;
            }

            const job = this.createJobCard();
            this.startInference(job, formData);
        },

        async validateForm() {
            const $audioPath = $('#audio_path');
            const $beatmapPath = $('#beatmap_path');
            const $outputPath = $('#output_path');

            const audioPath = $audioPath.val().trim() || $audioPath.attr('placeholder');
            const beatmapPath = $beatmapPath.val().trim() || $beatmapPath.attr('placeholder');
            const outputPath = $outputPath.val().trim() || $outputPath.attr('placeholder');

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
                ValidationManager.showInlineError('#beatmap_path', 'Must be .osu file');
                return false;
            }

            const pathsAreValid = await ValidationManager.validateAndAutofill(true);
            if (!pathsAreValid) {
                Utils.smoothScroll(0);
                return false;
            }

            return true;
        },

        createJobCard() {
            AppState.jobCounter += 1;
            const jobDisplayName = `Job ${AppState.jobCounter}`;
            const tempKey = `temp-${Date.now()}-${AppState.jobCounter}`;

            const $card = $(
                `<div class="progress-card" data-status="running" data-job-key="${tempKey}">
                    <div class="progress-card-header">
                        <div class="progress-card-title">${jobDisplayName}</div>
                        <button type="button" class="progress-card-close" title="Remove">×</button>
                    </div>
                    <div class="progress-card-status">Starting...</div>
                    <div class="warning-text" style="display:none; font-size: 12px; color: var(--accent-color); margin-top: 4px;">
                        Warning on this job (Will continue to generate)
                    </div>
                    <div class="init-message" style="font-style: italic; color: #ccc; margin-bottom: 10px;">
                        Initializing process... This may take a moment.
                    </div>
                    <div class="progressBarContainer">
                        <div class="progressBar"></div>
                    </div>
                    <div class="progress-card-actions">
                        <button type="button" class="cancel-button" style="display:none;">Cancel</button>
                    </div>
                    <div class="progress-card-links warning-log-link" style="display:none;">
                        <a href="#">View warning log</a>
                    </div>
                    <pre class="warning-log" style="display:none; white-space: pre-wrap; background: #141414; border: 1px solid var(--border-color); padding: 8px; border-radius: 6px; margin-top: 8px;"></pre>
                    <div class="progress-card-links beatmap-link" style="display:none;">
                        <a href="#" target="_blank">Click here to open the folder containing your map.</a>
                    </div>
                    <div class="progress-card-links error-log-link" style="display:none;">
                        <a href="#">See why... (opens error log)</a>
                    </div>
                </div>`
            );

            $('#progressCards').prepend($card);
            $('#progress_output').show();
            Utils.smoothScroll('#progress_output');

            const job = {
                id: null,
                tempKey,
                displayName: jobDisplayName,
                stage: 'starting',
                errorIndicatorSeen: false,
                warningMessages: [],
                warningCaptureActive: false,
                warningCaptureRemaining: 0,
                warningSuppressed: false,
                evtSource: null,
                isCancelled: false,
                inferenceErrorOccurred: false,
                accumulatedErrorMessages: [],
                errorLogFilePath: null,
                elements: {
                    $card,
                    $status: $card.find('.progress-card-status'),
                    $warningText: $card.find('.warning-text'),
                    $initMessage: $card.find('.init-message'),
                    $progressBar: $card.find('.progressBar'),
                    $progressBarContainer: $card.find('.progressBarContainer'),
                    $cancelButton: $card.find('.cancel-button'),
                    $warningLogLink: $card.find('.warning-log-link'),
                    $warningLogLinkAnchor: $card.find('.warning-log-link a'),
                    $warningLog: $card.find('.warning-log'),
                    $beatmapLink: $card.find('.beatmap-link'),
                    $beatmapLinkAnchor: $card.find('.beatmap-link a'),
                    $errorLogLink: $card.find('.error-log-link'),
                    $errorLogLinkAnchor: $card.find('.error-log-link a')
                }
            };

            $card.find('.progress-card-close').on('click', () => this.requestClose(job, $card));
            job.elements.$cancelButton.on('click', () => this.requestCancel(job));

            AppState.jobs.set(tempKey, job);
            return job;
        },

        shouldDelayStart() {
            if (!AppState.lastStartedJobId) return false;
            const lastJob = this.getJob(AppState.lastStartedJobId);
            if (!lastJob) return false;
            return lastJob.stage !== 'generating' && lastJob.stage !== 'finished';
        },

        removeJob(jobId, $cardOverride = null) {
            const job = this.getJob(jobId);
            const $card = $cardOverride || job?.elements?.$card;
            if (job?.evtSource) {
                job.evtSource.close();
            }
            if ($card) {
                const tempKey = $card.data('job-key');
                $card.remove();
                if (!jobId && tempKey) {
                    AppState.jobs.delete(tempKey);
                }
            }
            if (job) {
                AppState.jobs.delete(job.id || job.tempKey);
            }
            if (AppState.lastStartedJobId && job && AppState.lastStartedJobId === job.id) {
                AppState.lastStartedJobId = null;
            }
            this.updateProgressOutputVisibility();
        },

        removeFinishedCards() {
            $('.progress-card').each((_, card) => {
                const $card = $(card);
                const status = $card.data('status');
                if (status === 'completed' || status === 'error' || status === 'cancelled') {
                    const jobId = $card.data('job-id');
                    const tempKey = $card.data('job-key');
                    $card.remove();
                    if (jobId) {
                        AppState.jobs.delete(jobId);
                    } else if (tempKey) {
                        AppState.jobs.delete(tempKey);
                    }
                }
            });
            this.updateProgressOutputVisibility();
        },

        updateProgressOutputVisibility() {
            if ($('#progressCards').children().length === 0) {
                $('#progress_output').hide();
            }
        },

        buildFormData() {
            const formData = new FormData($("#inferenceForm")[0]);

            // Handle descriptors
            formData.delete('descriptors');
            const positiveDescriptors = [];
            const negativeDescriptors = [];

            $('input[name="descriptors"]').each(function() {
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

        startInference(job, formData) {
            $.ajax({
                url: "/start_inference",
                method: "POST",
                data: formData,
                processData: false,
                contentType: false,
                success: (response) => {
                    const jobId = response.job_id;
                    if (!jobId) {
                        Utils.showFlashMessage("Failed to start inference: missing job id.", 'error');
                        this.removeJob(job.id || job.tempKey, job.elements.$card);
                        return;
                    }
                    job.id = jobId;
                    job.elements.$cancelButton.show().prop('disabled', false).text('Cancel');
                    job.elements.$card.attr('data-job-id', jobId);
                    AppState.jobs.delete(job.tempKey);
                    AppState.jobs.set(jobId, job);
                    AppState.lastStartedJobId = jobId;
                    this.connectToSSE(job);
                },
                error: (jqXHR, textStatus, errorThrown) => {
                    console.error("Failed to start inference:", textStatus, errorThrown);
                    let errorMsg = "Failed to start inference process. Check backend console.";
                    if (jqXHR.responseJSON && jqXHR.responseJSON.message) {
                        errorMsg = jqXHR.responseJSON.message;
                    } else if (jqXHR.responseText) {
                        try {
                           const parsed = JSON.parse(jqXHR.responseText);
                           if(parsed && parsed.message) errorMsg = parsed.message;
                        } catch(e) { /* ignore parsing error */ }
                    }
                    Utils.showFlashMessage(errorMsg, 'error');
                    this.removeJob(job.id, job.elements.$card);
                }
            });
        },

        connectToSSE(job) {
            console.log("Connecting to SSE stream...", job.id);
            job.evtSource = new EventSource(`/stream_output?job_id=${encodeURIComponent(job.id)}`);
            job.errorLogFilePath = null;

            job.evtSource.onmessage = (e) => this.handleSSEMessage(job, e);
            job.evtSource.onerror = (err) => this.handleSSEError(job, err);
            job.evtSource.addEventListener("error_log", (e) => {
                job.errorLogFilePath = e.data;
            });
            job.evtSource.addEventListener("end", (e) => this.handleSSEEnd(job, e));
        },

        handleSSEMessage(job, e) {
            if (job.elements.$initMessage.is(":visible")) job.elements.$initMessage.hide();
            if (job.isCancelled) return;

            const messageData = e.data;
            const errorIndicators = [
                "Traceback (most recent call last):", "Error executing job with overrides:",
                "FileNotFoundError:", "Exception:", "Set the environment variable HYDRA_FULL_ERROR=1"
            ];

            const isErrorMessage = errorIndicators.some(indicator => messageData.includes(indicator));
            const isClientDisconnectTrace = messageData.includes("_client_handler") ||
                messageData.includes("Exception in thread Thread-") ||
                messageData.includes("GeneratorExit") ||
                messageData.includes("connection_dropped") ||
                messageData.includes("generator ignored GeneratorExit") ||
                messageData.includes("BrokenPipeError") ||
                messageData.includes("The pipe is being closed") ||
                messageData.includes("[WinError 232]");

            if (job.warningCaptureActive) {
                job.warningMessages.push(messageData);
                job.warningCaptureRemaining -= 1;
                if (job.warningCaptureRemaining <= 0) {
                    job.warningCaptureActive = false;
                }
                if (isClientDisconnectTrace) {
                    job.warningSuppressed = true;
                }
                if (job.warningSuppressed) {
                    job.warningMessages = [];
                    job.warningCaptureActive = false;
                    job.elements.$warningLog.hide().text('');
                    job.elements.$warningLogLink.hide();
                    job.elements.$warningText.hide();
                    return;
                }
                job.elements.$warningLog.text(job.warningMessages.join("\n"));
            }

            if (isErrorMessage && !isClientDisconnectTrace) {
                job.errorIndicatorSeen = true;
                job.accumulatedErrorMessages.push(messageData);
                job.warningMessages.push(messageData);
                job.warningCaptureActive = true;
                job.warningCaptureRemaining = 80;
                job.elements.$warningText.show();
                job.elements.$warningLog.text(job.warningMessages.join("\n"));
                if (job.elements.$warningLogLink.is(':hidden')) {
                    job.elements.$warningLogLinkAnchor.off("click").on("click", (event) => {
                        event.preventDefault();
                        job.elements.$warningLog.toggle();
                    });
                    job.elements.$warningLogLink.show();
                }
            } else if (job.inferenceErrorOccurred) {
                job.accumulatedErrorMessages.push(messageData);
            } else {
                this.updateProgress(job, messageData);
            }
        },

        updateProgress(job, messageData) {
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
                    job.elements.$status.text(title);
                    if (job.stage !== 'generating') {
                        job.stage = 'generating';
                    }
                }
            });

            // Update progress bar
            const progressMatch = messageData.match(/^\s*(\d+)%\|/);
            if (progressMatch) {
                const currentPercent = parseInt(progressMatch[1].trim(), 10);
                if (!isNaN(currentPercent)) {
                    job.elements.$progressBar.css("width", currentPercent + "%");
                }
            }

            // Check for completion message
            if (messageData.includes("Generated beatmap saved to")) {
                const parts = messageData.split("Generated beatmap saved to");
                if (parts.length > 1) {
                    const fullPath = parts[1].trim().replace(/\\/g, "/");
                    const folderPath = fullPath.substring(0, fullPath.lastIndexOf("/"));

                    job.elements.$beatmapLinkAnchor
                        .attr("href", "#")
                        .text("Click here to open the folder containing your map.")
                        .off("click")
                        .on("click", (e) => {
                            e.preventDefault();
                            $.get("/open_folder", { folder: folderPath })
                                .done(response => console.log("Open folder response:", response))
                                .fail(() => alert("Failed to open folder via backend."));
                        });
                    job.elements.$beatmapLink.show();
                }
            }
        },

        handleSSEError(job, err) {
            console.error("EventSource failed:", err);
            if (job.evtSource) {
                job.evtSource.close();
                job.evtSource = null;
            }

            job.stage = 'finished';

            if (!job.isCancelled && !job.inferenceErrorOccurred) {
                job.inferenceErrorOccurred = true;
                job.accumulatedErrorMessages.push("Error: Connection to process stream lost.");
                job.elements.$status.text("Connection Error").css('color', 'var(--accent-color)');
                job.elements.$progressBar.addClass('error');
                job.elements.$card.data('status', 'error');
                Utils.showFlashMessage("Error: Connection to process stream lost.", "error");
            }

            job.elements.$cancelButton.hide();
        },

        handleSSEEnd(job, e) {
            console.log("Received end event from server.", e.data);
            if (job.evtSource) {
                job.evtSource.close();
                job.evtSource = null;
            }

            const endMessage = (e.data || '').toLowerCase();
            const endWithErrors = endMessage.includes('with errors');
            if (endWithErrors) {
                job.inferenceErrorOccurred = true;
            }

            if (job.isCancelled) {
                job.elements.$status.text("Cancelled").css('color', 'var(--accent-color)');
                job.elements.$progressBar.addClass('error');
                job.elements.$card.data('status', 'cancelled');
            } else if (job.inferenceErrorOccurred) {
                this.handleInferenceError(job);
                job.elements.$card.data('status', 'error');
            } else {
                job.elements.$status.text("Processing Complete").css('color', '');
                job.elements.$progressBar.css("width", "100%").removeClass('error');
                job.elements.$card.data('status', 'completed');
            }

            job.elements.$cancelButton.hide();
            job.isCancelled = false;
        },

        handleInferenceError(job) {
            const fullErrorText = job.accumulatedErrorMessages.join("\\n");
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
            job.elements.$status.text("Processing Failed").css('color', 'var(--accent-color)').show();
            job.elements.$progressBar.css("width", "100%").addClass('error');
            job.elements.$beatmapLink.hide();

            if (job.errorLogFilePath) {
                job.elements.$errorLogLinkAnchor.off("click").on("click", (e) => {
                    e.preventDefault();
                    $.get("/open_log_file", { path: job.errorLogFilePath })
                        .done(response => console.log("Open log response:", response))
                        .fail(() => alert("Failed to open log file via backend."));
                });
                job.elements.$errorLogLink.show();
            }
        },

        requestCancel(job) {
            if (job.stage !== 'generating' && job.stage !== 'finished') {
                Utils.showFlashMessage(`Please wait until ${job.displayName} starts generating before cancelling.`, 'error');
                return;
            }
            this.scheduleCancelClose(() => this.cancelInference(job));
        },

        cancelInference(job) {
            const $cancelBtn = job.elements.$cancelButton;
            $cancelBtn.prop('disabled', true).text('Cancelling...');

            $.ajax({
                url: "/cancel_inference",
                method: "POST",
                data: { job_id: job.id },
                success: (response) => { // Expecting JSON response
                    job.isCancelled = true;
                    Utils.showFlashMessage(response.message || "Inference cancelled successfully.", "cancel-success");
                },
                error: (jqXHR) => {
                    const errorMsg = jqXHR.responseJSON?.message || "Failed to send cancel request. Unknown error.";
                    Utils.showFlashMessage(errorMsg, "error");
                    $cancelBtn.prop('disabled', false).text('Cancel');
                }
            });
        },

        requestClose(job, $card) {
            const status = $card?.data('status');
            if (job.stage === 'finished' || status === 'completed' || status === 'error' || status === 'cancelled') {
                this.removeJob(job.id || job.tempKey, $card);
                return;
            }
            if (job.stage !== 'generating' && job.stage !== 'finished') {
                Utils.showFlashMessage(`Please wait until ${job.displayName} starts generating before cancelling.`, 'error');
                return;
            }
            this.scheduleCancelClose(() => {
                this.cancelInference(job);
                this.removeJob(job.id || job.tempKey, $card);
            });
        },

        scheduleCancelClose(action) {
            const now = Date.now();
            if (now < AppState.nextCancelCloseAt) {
                const waitMs = AppState.nextCancelCloseAt - now;
                Utils.showFlashMessage(`Please wait ${Math.ceil(waitMs / 1000)}s before trying again.`, 'error');
                return;
            }

            action();
            AppState.nextCancelCloseAt = now + AppState.cancelCloseCooldownMs;
        },

        getJob(jobId) {
            return AppState.jobs.get(jobId) || null;
        }
    };

    // Initialize all components
    function initializeApp() {
        // Initialize language selector
        const $langSelector = $('#language-selector');
        if ($langSelector.length && typeof I18n !== 'undefined') {
            // Set initial value from I18n
            const currentLang = I18n.getCurrentLanguage();
            $langSelector.val(currentLang);
            
            // Handle language change
            $langSelector.on('change', async function() {
                const newLang = $(this).val();
                const success = await I18n.setLanguage(newLang);
                if (!success) {
                    // Revert to current language if failed
                    $(this).val(I18n.getCurrentLanguage());
                }
            });
        }

        // Check BF16 support on page load
        $.get("/check_bf16_support", function(data) {
            if (data.supported) {
                $("#bf16-option").show();
                if (data.gpu_name) {
                    $("#bf16-gpu-info").text("(" + data.gpu_name + ")");
                }
            }
        });

        // Initialize Select2
        $('.select2').select2({
            placeholder: "Select options",
            allowClear: true,
            dropdownCssClass: "select2-dropdown-dark",
            containerCssClass: "select2-container-dark"
        });

        // Initialize all managers
        FileBrowser.init();
        UIManager.init();
        ValidationManager.init();
        DescriptorManager.init();
        ConfigManager.init();
        InferenceManager.init();

        // Attach event handlers
        $("#model").on('change', () => UIManager.updateModelSettings());
        $("#gamemode").on('change', () => UIManager.updateConditionalFields());

        // Initial UI updates
        UIManager.updateModelSettings();
    }

    // Start the application
    initializeApp();
});
