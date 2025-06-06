$(document).ready(function() {
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
            $('#export_osz, #add_to_beatmap, #super_timing').prop('checked', false);

            // Clear descriptors and context options
            $('input[name="descriptors"], input[name="in_context_options"]')
                .removeClass('positive-check negative-check').prop('checked', false);

            // Clear paths and optional fields
            $('#audio_path, #output_path, #beatmap_path, #mapper_id, #seed, #start_time, #end_time, #hold_note_ratio, #scroll_speed_ratio').val('');
        }
    };

    // UI Manager for conditional visibility
    const UIManager = {
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
            ['#in-context-options-box', '#add-to-beatmap-option'].forEach(selector => {
                const $element = $(selector);
                if (shouldShowBeatmapFields && !$element.is(':visible')) {
                    $element.fadeIn(AppState.animationSpeed);
                } else if (!shouldShowBeatmapFields && $element.is(':visible')) {
                    $element.fadeOut(AppState.animationSpeed);
                    if (selector === '#add-to-beatmap-option') {
                        $('#add_to_beatmap').prop('checked', false);
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
            window.addEventListener('pywebviewready', () => {
                console.log("pywebview API is ready.");
                this.attachBrowseHandlers();
            });
        },

        attachBrowseHandlers() {
            $('.browse-button[data-browse-type]').click(async function() {
                const browseType = $(this).data('browse-type');
                const targetId = $(this).data('target');

                try {
                    const path = browseType === 'folder' ?
                                await window.pywebview.api.browse_folder() :
                                await window.pywebview.api.browse_file();

                    if (path) {
                        $(`#${targetId}`).val(path).trigger('input');
                        console.log(`Selected ${browseType}:`, path);
                    }
                } catch (error) {
                    console.error(`Error browsing for ${browseType}:`, error);
                    alert(`Could not browse for ${browseType}. Ensure the backend API is running.`);
                }
            });
        }
    };

    // Descriptor Manager
    const DescriptorManager = {
        init() {
            this.attachDropdownHandler();
            this.attachDescriptorClickHandlers();
        },

        attachDropdownHandler() {
            $('.custom-dropdown-descriptors .dropdown-header').click(function() {
                const $dropdown = $(this).parent();
                $dropdown.toggleClass('open');
                if ($dropdown.hasClass('open')) {
                    Utils.smoothScroll(this);
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

            if (window.pywebview?.api?.browse_folder) {
                this.exportToFolder(config);
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

        async exportToFolder(config) {
            try {
                const folderPath = await window.pywebview.api.browse_folder();
                if (!folderPath) {
                    this.showConfigStatus("Export cancelled by user", "error");
                    return;
                }

                const filename = `mapperatorinator-config-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;

                $.ajax({
                    url: "/save_config",
                    method: "POST",
                    data: {
                        folder_path: folderPath,
                        filename: filename,
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

                // Store current state
                const currentBeatmapPath = $('#beatmap_path').val().trim();

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

                const newBeatmapPath = $('#beatmap_path').val().trim();
                if (newBeatmapPath !== currentBeatmapPath) {
                    $("#beatmap_path").trigger('input');
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
        init() {
            $('#inferenceForm').submit((e) => this.handleSubmit(e));
            $('#cancel-button').click(() => this.cancelInference());
        },

        handleSubmit(e) {
            e.preventDefault();

            if (!this.validateForm()) return;

            this.resetProgress();
            this.startInference();
        },

        validateForm() {
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

            // Ensure hitsounded for V30
            if ($("#model").val() === "v30" && !$("#option-item-hitsounded").is(':visible')) {
                formData.set('hitsounded', 'true');
            }

            return formData;
        },

        startInference() {
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
                    alert("Failed to start inference process. Check backend console.");
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
            } else if (AppState.inferenceErrorOccurred) {
                this.handleInferenceError();
            } else {
                $("#progressTitle").show().text("Processing Complete").css('color', '');
                $("#progressBarContainer").show();
                $("#progressBar").css("width", "100%").removeClass('error');
            }

            $("button[type='submit']").prop("disabled", false);
            $("#cancel-button").hide();
            AppState.isCancelled = false;
            AppState.inferenceErrorOccurred = false;
            AppState.accumulatedErrorMessages = [];
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
                success: () => {
                    AppState.isCancelled = true;
                    $("#progress_output").hide();
                    Utils.showFlashMessage("Inference cancelled successfully.", "cancel-success");
                },
                error: (jqXHR) => {
                    const errorMsg = jqXHR.responseJSON?.message || "Failed to send cancel request. Unknown error.";
                    Utils.showFlashMessage(errorMsg, "error");
                    $cancelBtn.prop('disabled', false).text('Cancel');
                }
            });
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
        DescriptorManager.init();
        ConfigManager.init();
        InferenceManager.init();

        // Attach event handlers
        $("#model").on('change', () => UIManager.updateModelSettings());
        $("#gamemode").on('change', () => UIManager.updateConditionalFields());
        $("#beatmap_path").on('input', () => UIManager.updateConditionalFields());

        // Initial UI updates
        UIManager.updateModelSettings();
    }

    // Start the application
    initializeApp();
});
