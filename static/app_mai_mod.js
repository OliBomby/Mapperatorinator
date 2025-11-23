$(document).ready(function() {
    // Application state and configuration
    const AppState = {
        evtSource: null,
        isCancelled: false,
        inferenceErrorOccurred: false,
        printingSuggestions: false,
        accumulatedSuggestions: [],
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

            // Clear paths and optional fields
            $('#beatmap_path').val('');
            PathManager.validateAndAutofillPaths(false);
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
            $('#beatmap_path').trigger('blur');
        },

        attachPathChangeHandlers() {
            // Listen for input events (typing)
            $('#beatmap_path').on('input', (e) => {
                this.updateClearButtonVisibility(e.target);
            });

            // Listen for blur events (leaving field) - immediate validation
            $('#beatmap_path').on('blur', (e) => {
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
            $('#beatmap_path').each((index, element) => {
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
            const beatmapPath = $('#beatmap_path').val().trim();

            // Only validate if at least one path is provided
            if (!beatmapPath) {
                return Promise.resolve(true);
            }

            // Call backend validation
            return new Promise((resolve) => {
                $.ajax({
                    url: '/validate_paths',
                    method: 'POST',
                    data: {
                        beatmap_path: beatmapPath,
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
        },

        showInlineErrorForMessage(error) {
            const beatmapPathVal = $('#beatmap_path').val().trim();

            if (error.includes('Audio file not found') && ( beatmapPathVal)) {
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
    };

    // Inference Manager
    const InferenceManager = {
        init() {
            $('#inferenceForm').submit((e) => this.handleSubmit(e));
            $('#cancel-button').click(() => this.cancelInference());

            // Add a single click event handler for all section headers (event delegation)
            $('#output-container').on('click', '.section-header', function() {
                $(this).closest('.issue-section').toggleClass('collapsed');
            });
        },

        async handleSubmit(e) {
            e.preventDefault();

            // Apply placeholder values before validation
            if (!await this.validateForm()) return;

            this.resetProgress();
            this.startInference();
        },

        async validateForm() {
            const beatmapPath = $('#beatmap_path').val().trim();

            if (!beatmapPath) {
                Utils.smoothScroll(0);
                Utils.showFlashMessage("'Beatmap Path' is required for running MaiMod", 'error');
                return false;
            }

            // Validate beatmap file type if beatmap path is provided
            if (!beatmapPath.toLowerCase().endsWith('.osu')) {
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
            AppState.printingSuggestions = false;
            AppState.accumulatedSuggestions = [];
            AppState.accumulatedErrorMessages = [];
            AppState.isCancelled = false;

            if (AppState.evtSource) {
                AppState.evtSource.close();
                AppState.evtSource = null;
            }
        },

        buildFormData() {
            return new FormData($("#inferenceForm")[0]);
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
            } else if (AppState.printingSuggestions){
                AppState.accumulatedSuggestions.push(messageData);
            } else {
                this.updateProgress(messageData);
            }
        },

        updateProgress(messageData) {
            // Update progress title
            $("#progressTitle").text("Processing...");

            // Update progress bar
            const progressMatch = messageData.match(/^\s*(\d+)%\|/);
            if (progressMatch) {
                const currentPercent = parseInt(progressMatch[1].trim(), 10);
                if (!isNaN(currentPercent)) {
                    $("#progressBar").css("width", currentPercent + "%");
                }
            }

            // Check for completion message
            if (messageData.includes("suggestions:")) {
                AppState.printingSuggestions = true;
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
                $("#progress_output").hide();
            } else if (AppState.inferenceErrorOccurred) {
                this.handleInferenceError();
            } else {
                $("#progressTitle, #progressBarContainer, #beatmapLink, #errorLogLink").hide();
                $("#progress_output").hide();
                this.handleSuggestions();
            }

            $("button[type='submit']").prop("disabled", false);
            $("#cancel-button").hide();
            AppState.isCancelled = false;
        },

        handleSuggestions() {
            if (AppState.accumulatedSuggestions.length === 0) {
                Utils.showFlashMessage("No suggestions available.", "info");
                return;
            }

            const logData = AppState.accumulatedSuggestions;
            const container = $('#output-container');
            let currentSection = null;

            container.empty(); // Clear previous content

            // Regex to parse an issue line, now capturing any color name.
            const issueRegex = /^\s*(?:\[bold(?:\s+([a-zA-Z]+))?\])?\((\d+)\)(?:\[\/bold.*?\])?\s*\[link=([^\]]+)\]\[green\]([^\]]+)\[\/green\]\[\/link\]\s*(\([^\)]+\))\s*-\s*(.*)$/;

            // Function to create a new section
            const createSection = (title) => {
                const section = $(`
                    <div class="issue-section">
                        <div class="section-header">
                            <h2>${title}</h2>
                            <span class="toggle-btn">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640"><path d="M297.4 470.6C309.9 483.1 330.2 483.1 342.7 470.6L534.7 278.6C547.2 266.1 547.2 245.8 534.7 233.3C522.2 220.8 501.9 220.8 489.4 233.3L320 402.7L150.6 233.4C138.1 220.9 117.8 220.9 105.3 233.4C92.8 245.9 92.8 266.2 105.3 278.7L297.3 470.7z"/></svg>
                            </span>
                        </div>
                        <div class="issue-list">
                            <div class="issue-list-content">
                                </div>
                        </div>
                    </div>
                `);
                container.append(section);
                return section;
            };

            // Main processing loop
            logData.forEach(line => {
                line = line.trim();
                if (!line) return; // Skip empty lines

                if (line.endsWith(':')) {
                    const title = line.slice(0, -1);
                    currentSection = createSection(title);
                } else {
                    const match = line.match(issueRegex);
                    if (match) {
                        // Deconstruct the matched parts from the regex
                        let [, color, severity, link, timestamp, object, description] = match;

                        // Set a default color if none is specified
                        if (!color) {
                            color = '#89b4fa'; // Default color (a light blue from the palette)
                        } else if (color === 'yellow') {
                            color = '#f7fc76'; // Yellow color
                        } else if (color === 'red') {
                            color = '#f87171'; // Red color
                        }

                        // Parse links inside the description text
                        const formattedDescription = description.replace(
                            /\[link=([^\]]+)\](.*?)\[\/link\]/g,
                            '<a href="$1" class="issue-timestamp">$2</a>'
                        );

                        if (!currentSection) {
                            currentSection = createSection('General');
                        }

                        // Find a fitting font size based on severity digit count so it fits well in the circle
                        // Remove the last character from severity because we dont need the precision
                        const severityDigitCount = severity.length;
                        let fontSize;
                        if (severityDigitCount <= 2) {
                            fontSize = '16px'; // Single digit
                        } else if (severityDigitCount === 3) {
                            fontSize = '14.4px'; // Double digits
                        } else {
                            fontSize = '11px'; // More than two digits
                        }

                        // Build the issue item with the new structure
                        const issueItem = $(`
                            <div class="issue-item">
                                <div class="severity-circle" style="background-color: ${color}; font-size: ${fontSize};"" title="The importance of the suggestion. Values above 100 are likely issues, whereas values below 10 are likely subjective.">${severity}</div>
                                <a href="${link}" class="issue-timestamp">${timestamp}</a>
                                <span class="issue-object">${object}</span>
                                <span class="issue-description">- ${formattedDescription}</span>
                            </div>
                        `);

                        currentSection.find('.issue-list-content').append(issueItem);
                    }
                }
            });

            $("#output-container").show();
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
        InferenceManager.init();
    }

    // Start the application
    initializeApp();
});
