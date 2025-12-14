/**
 * queue_manager.js – Generation Queue & Mapper List Management
 * 
 * Features:
 * - Multi-map batch processing queue
 * - Mapper style sampling with quantities  
 * - Queue visualization with status indicators
 * - Async queue processing that doesn't block UI
 */

/* ================================================================== */
/*  HELPER FUNCTIONS                                                  */
/* ================================================================== */

function _defaultDiffLabel(stars) {
    if (stars < 2.0) return "Easy";
    else if (stars < 2.7) return "Normal";
    else if (stars < 4.0) return "Hard";
    else if (stars < 5.3) return "Insane";
    else if (stars < 6.5) return "Expert";
    else return "Expert+";
}

/* ================================================================== */
/*  QUEUE MANAGER                                                     */
/* ================================================================== */

const QueueManager = (() => {
    const queue = [];
    let running = false;
    let cancelled = false;
    let currentTaskId = 0;

    /**
     * Render the queue list in the UI
     */
    function render() {
        const list = document.getElementById("queue-list");
        if (!list) return;

        list.innerHTML = "";

        if (queue.length === 0) {
            list.innerHTML = '<li class="queue-empty">Queue is empty</li>';
            return;
        }

        queue.forEach((task, idx) => {
            const li = document.createElement("li");
            li.classList.add("queue-item");
            li.dataset.taskId = task.id;

            // Status indicator
            const isRunning = idx === 0 && running;
            const statusClass = isRunning ? "status-running" : "status-pending";
            const statusSymbol = isRunning ? "●" : "○";
            const statusText = isRunning ? "Running" : "Pending";

            // Build item content
            const statusSpan = document.createElement("span");
            statusSpan.className = statusClass;
            statusSpan.textContent = `${statusSymbol} ${statusText}`;

            const nameSpan = document.createElement("span");
            nameSpan.className = "task-name";
            nameSpan.textContent = task.display_name || "Unnamed Task";
            nameSpan.title = task.display_name;

            const deleteBtn = document.createElement("button");
            deleteBtn.type = "button";
            deleteBtn.className = "queue-delete-btn";
            deleteBtn.textContent = "×";
            deleteBtn.title = isRunning ? "Skip current task" : "Remove from queue";
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                remove(idx);
            };

            li.appendChild(statusSpan);
            li.appendChild(nameSpan);
            li.appendChild(deleteBtn);

            if (isRunning) {
                li.classList.add("running");
            }

            list.appendChild(li);
        });
    }

    /**
     * Add a task to the queue
     */
    function add(task) {
        task.id = ++currentTaskId;
        queue.push(task);
        // Don't call render() - let app.js handle UI updates via updateUI()
        console.log("[Queue] Added task:", task.display_name);
    }

    /**
     * Clear all pending tasks (preserves running task)
     */
    function clear() {
        if (running && queue.length > 0) {
            // Keep the currently running task
            const current = queue[0];
            queue.length = 0;
            queue.push(current);
        } else {
            queue.length = 0;
        }
        // Don't call render() - let app.js handle UI updates
    }

    /**
     * Stop the queue completely and cancel current task
     */
    function stop() {
        cancelled = true;
        running = false;
        queue.length = 0;
        // Don't call render() - let app.js handle UI updates
        console.log("[Queue] Stopped");
    }

    /**
     * Remove a task from the queue by index
     */
    async function remove(index) {
        if (index < 0 || index >= queue.length) return;

        if (running && index === 0) {
            // Skip current task - cancel it and move to next
            console.log("[Queue] Skipping current task");
            try {
                await fetch("/cancel_inference", { method: "POST" });
            } catch (e) {
                console.error("[Queue] Cancel request failed:", e);
            }
            // The task will be removed when it completes/fails
            // and _runNext will be called
        } else {
            // Just remove the pending task
            queue.splice(index, 1);
            // UI update handled by app.js
        }
    }

    /**
     * Check if there are pending tasks
     */
    function hasPending() {
        return queue.length > 0;
    }

    /**
     * Check if queue is currently processing
     */
    function isRunning() {
        return running;
    }

    /**
     * Internal: Run the next task in queue
     */
    async function _runNext() {
        if (cancelled || queue.length === 0) {
            running = false;
            cancelled = false;
            // Trigger UI update via custom event
            window.dispatchEvent(new CustomEvent('queueStateChanged'));
            return;
        }

        running = true;
        const task = queue[0];
        // Trigger UI update via custom event
        window.dispatchEvent(new CustomEvent('queueStateChanged'));

        console.log("[Queue] Starting task:", task.display_name);

        try {
            window._queueInProgress = true;
            await InferenceRunner.runTask(task);
        } catch (e) {
            console.error("[Queue] Task failed:", e);
            if (typeof Utils !== 'undefined' && Utils.showFlashMessage) {
                Utils.showFlashMessage(`Task failed: ${task.display_name}`, 'error');
            }
        } finally {
            // Remove completed task
            if (queue.length > 0 && queue[0].id === task.id) {
                queue.shift();
            }
            window._queueInProgress = false;

            // Continue to next task if not cancelled
            if (!cancelled && queue.length > 0) {
                // Small delay to allow UI to update
                setTimeout(() => _runNext(), 500);
            } else {
                running = false;
                cancelled = false;
                // Trigger UI update via custom event
                window.dispatchEvent(new CustomEvent('queueStateChanged'));
            }
        }
    }

    /**
     * Start processing the queue
     */
    function start() {
        if (!running && queue.length > 0) {
            cancelled = false;
            _runNext();
        }
    }

    /**
     * Get queue length
     */
    function getLength() {
        return queue.length;
    }

    /**
     * Check if queue is empty
     */
    function isEmpty() {
        return queue.length === 0;
    }

    /**
     * Get all tasks in the queue
     */
    function getAllTasks() {
        return queue.map(task => ({
            id: task.id,
            formData: task,
            status: task.status || 'pending'  // Use the status set by setTaskStatus, or 'pending' as default
        }));
    }

    /**
     * Set task status (for UI updates)
     */
    function setTaskStatus(taskId, status) {
        const task = queue.find(t => t.id === taskId);
        if (task) {
            task.status = status;
            // Trigger UI update via custom event
            window.dispatchEvent(new CustomEvent('queueStateChanged'));
        }
    }

    /**
     * Remove a task by ID
     */
    function removeTask(taskId) {
        // Convert to number for comparison since task.id is a number
        const numericId = typeof taskId === 'string' ? parseInt(taskId, 10) : taskId;
        const index = queue.findIndex(t => t.id === numericId);
        if (index !== -1) {
            remove(index);
        }
    }

    return {
        add,
        addTask: add,  // Alias for app.js compatibility
        clear,
        clearAll: clear,  // Alias for app.js compatibility
        remove,
        removeTask,  // By ID version
        stop,
        hasPending,
        isRunning,
        isEmpty,
        getAllTasks,
        setTaskStatus,
        start,
        render,
        getLength
    };
})();


/* ================================================================== */
/*  MAPPER LIST MANAGER                                               */
/* ================================================================== */

const MapperManager = (() => {
    const listEl = document.getElementById("mappers-list");
    if (!listEl) {
        console.warn("[MapperManager] mappers-list element not found");
        return {
            addMapper: () => { },
            gatherSelected: () => [],
            getAll: () => [],
            clearAll: () => { },
            loadFromArray: () => { }
        };
    }

    /**
     * Generate HTML for a mapper entry
     */
    function _entryTemplate(id, name, count = 1, checked = true) {
        return `
        <div class="mapper-item" data-id="${id}">
            <input type="checkbox" class="mapper-check" ${checked ? 'checked' : ''} title="Include in queue generation">
            <span class="mapper-id">#${id}</span>
            <input type="text" class="mapper-name" value="${name}" title="Display name">
            <label class="count-label">×</label>
            <input type="number" class="mapper-count" value="${count}" min="1" max="99" title="Number of maps to generate">
            <button type="button" class="add-single-mapper-btn" title="Add this mapper to queue">+</button>
            <button type="button" class="remove-mapper-btn" title="Remove mapper">×</button>
        </div>`;
    }

    /**
     * Update the empty state message visibility
     */
    function _updateEmptyState() {
        const items = listEl.querySelectorAll(".mapper-item");
        const emptyMsg = listEl.querySelector(".mapper-empty");

        if (items.length === 0) {
            if (!emptyMsg) {
                listEl.innerHTML = '<div class="mapper-empty">No mappers added. Add mappers to auto-generate queue tasks.</div>';
            }
        } else if (emptyMsg) {
            emptyMsg.remove();
        }
    }

    /**
     * Add a mapper by ID
     * @param {string} id - Mapper ID
     * @param {string} name - Optional mapper name (if not provided, fetches from API)
     * @param {number} count - Number of maps to generate (default 1)
     */
    async function addMapper(id, name = null, count = 1) {
        id = String(id).trim();
        if (!id || !/^\d+$/.test(id)) {
            if (typeof Utils !== 'undefined') {
                Utils.showFlashMessage("Please enter a valid mapper ID (numbers only)", "error");
            }
            return;
        }

        // Remove empty state message if present
        const emptyMsg = listEl.querySelector(".mapper-empty");
        if (emptyMsg) emptyMsg.remove();

        // Check for duplicates
        const existing = listEl.querySelector(`.mapper-item[data-id="${id}"]`);
        if (existing) {
            if (typeof Utils !== 'undefined') {
                Utils.showFlashMessage("Mapper already added", "error");
            }
            return;
        }

        // If name provided, use it directly
        if (name && name !== 'Unknown') {
            listEl.insertAdjacentHTML("beforeend", _entryTemplate(id, name, count));
            _updateEmptyState();
            return;
        }

        // Show loading state and fetch name
        listEl.insertAdjacentHTML("beforeend", _entryTemplate(id, "Loading…", count));
        const newItem = listEl.lastElementChild;

        // Fetch username from API
        try {
            const res = await fetch("/lookup_mapper_name", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mapper_id: id })
            });

            if (res.ok) {
                const data = await res.json();
                name = data.username || `ID ${id}`;
            } else {
                name = `ID ${id}`;
            }
        } catch (e) {
            console.error("[MapperManager] Lookup failed:", e);
            name = `ID ${id}`;
        }

        // Update the name field
        const nameInput = newItem.querySelector(".mapper-name");
        if (nameInput) {
            nameInput.value = name;
        }
    }

    /**
     * Get all mappers (for config export)
     */
    function getAll() {
        return Array.from(listEl.querySelectorAll(".mapper-item")).map(div => ({
            id: div.dataset.id,
            name: div.querySelector(".mapper-name")?.value.trim() || "",
            n: parseInt(div.querySelector(".mapper-count")?.value, 10) || 1,
            checked: div.querySelector(".mapper-check")?.checked ?? true
        }));
    }

    /**
     * Clear all mappers
     */
    function clearAll() {
        listEl.innerHTML = '<div class="mapper-empty">No mappers added. Add mappers to auto-generate queue tasks.</div>';
    }

    /**
     * Load mappers from array (for config import)
     */
    function loadFromArray(arr) {
        listEl.innerHTML = '';  // Clear first
        arr.forEach(obj => {
            const { id, name, n = 1, checked = true } = obj;
            listEl.insertAdjacentHTML("beforeend", _entryTemplate(id, name, n, checked));
        });
        _updateEmptyState();
    }

    /**
     * Get selected/checked mappers for queue generation
     */
    function gatherSelected() {
        return Array.from(listEl.querySelectorAll(".mapper-item"))
            .filter(div => div.querySelector(".mapper-check")?.checked)
            .map(div => ({
                id: div.dataset.id,
                name: div.querySelector(".mapper-name")?.value.trim() || "",
                n: parseInt(div.querySelector(".mapper-count")?.value, 10) || 1
            }));
    }

    /**
     * Get a single mapper by ID
     */
    function getMapper(id) {
        const div = listEl.querySelector(`.mapper-item[data-id="${id}"]`);
        if (!div) return null;
        return {
            id: div.dataset.id,
            name: div.querySelector(".mapper-name")?.value.trim() || "",
            n: parseInt(div.querySelector(".mapper-count")?.value, 10) || 1
        };
    }

    // Event delegation for remove and add-single buttons
    listEl.addEventListener("click", (e) => {
        if (e.target.classList.contains("remove-mapper-btn")) {
            e.target.closest(".mapper-item")?.remove();
            _updateEmptyState();
        }

        if (e.target.classList.contains("add-single-mapper-btn")) {
            const item = e.target.closest(".mapper-item");
            if (item && typeof window.QueueUI !== 'undefined') {
                const mapperId = item.dataset.id;
                const mapperName = item.querySelector(".mapper-name")?.value.trim() || "";
                const count = parseInt(item.querySelector(".mapper-count")?.value, 10) || 1;
                window.QueueUI.addSingleMapperToQueue(mapperId, mapperName, count);
            }
        }
    });

    return { addMapper, gatherSelected, getAll, getMapper, clearAll, loadFromArray };
})();


/* ================================================================== */
/*  INFERENCE RUNNER (Queue Task Executor)                            */
/* ================================================================== */

const InferenceRunner = {
    /**
     * Run a single task from the queue
     */
    async runTask(task) {
        return new Promise((resolve, reject) => {
            // Store resolver for SSE end event
            window._queueResolver = resolve;
            window._queueRejecter = reject;

            // Build form data from task
            const formData = new FormData();

            Object.entries(task).forEach(([key, value]) => {
                if (value !== undefined && value !== null && value !== '') {
                    // Handle special keys
                    if (key === 'mapper_id' && value) {
                        formData.set('mapper_id', value);
                    } else if (!['id', 'display_name', 'mapper_display_name'].includes(key)) {
                        formData.set(key, value);
                    }
                }
            });

            // Ensure mapper_id is included if present
            if (task.mapper_id) {
                formData.set('mapper_id', task.mapper_id);
            }

            // Submit via existing inference handler
            if (typeof window.startInferenceWithFormData === 'function') {
                window.startInferenceWithFormData(formData)
                    .then(resolve)
                    .catch(reject);
            } else {
                // Fallback: POST directly
                fetch("/start_inference", {
                    method: "POST",
                    body: formData
                })
                    .then(res => {
                        if (!res.ok) throw new Error(`HTTP ${res.status}`);
                        // Wait for SSE completion
                    })
                    .catch(reject);
            }
        });
    }
};


/* ================================================================== */
/*  UI EVENT BINDINGS                                                 */
/* ================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // Add mapper button
    const addMapperBtn = document.getElementById("add-mapper-btn");
    const mapperInput = document.getElementById("add-mapper-id");

    if (addMapperBtn && mapperInput) {
        addMapperBtn.onclick = () => {
            const id = mapperInput.value.trim();
            mapperInput.value = "";
            MapperManager.addMapper(id);
        };

        // Allow Enter key to add mapper
        mapperInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                addMapperBtn.click();
            }
        });
    }

    // Add to queue button
    const addToQueueBtn = document.getElementById("addToQueueBtn");
    if (addToQueueBtn) {
        addToQueueBtn.onclick = () => {
            // Validate paths first
            if (typeof PathManager !== 'undefined' && PathManager.applyPlaceholderValues) {
                PathManager.applyPlaceholderValues();
            }

            const audioPath = document.getElementById("audio_path")?.value.trim();
            const beatmapPath = document.getElementById("beatmap_path")?.value.trim();

            if (!audioPath && !beatmapPath) {
                if (typeof Utils !== 'undefined') {
                    Utils.showFlashMessage("Load an audio or beatmap file first", "error");
                }
                return;
            }

            // Get selected mappers
            const mappers = MapperManager.gatherSelected();
            const form = document.getElementById("inferenceForm");
            if (!form) return;

            const fd = new FormData(form);

            // If no mappers selected, add one task with current settings
            const mappersToUse = mappers.length > 0 ? mappers : [{ id: "", name: "", n: 1 }];

            let tasksAdded = 0;

            mappersToUse.forEach(mp => {
                for (let i = 0; i < mp.n; i++) {
                    // Build task from form data
                    const task = Object.fromEntries(fd.entries());

                    // Set mapper info
                    if (mp.id) {
                        task.mapper_id = mp.id;
                    }
                    task.mapper_display_name = mp.name || "";

                    // Build display name
                    const artist = task.artist || fd.get("song_artist") || "??";
                    const title = task.title || fd.get("song_title") || "??";
                    const model = fd.get("model") || "v30";

                    // Determine creator name
                    let creator = fd.get("mapper_name")?.trim();
                    if (!creator) {
                        creator = mp.name || `Mapperatorinator ${model.toUpperCase()}`;
                    }

                    // Determine difficulty name
                    let diffName = fd.get("difficulty_name")?.trim();
                    if (!diffName) {
                        const stars = parseFloat(fd.get("difficulty")) || 5.0;
                        const baseDiff = _defaultDiffLabel(stars);
                        if (mp.id || mp.name) {
                            diffName = `${creator}'s ${baseDiff}`;
                        } else {
                            diffName = baseDiff;
                        }
                    }

                    task.creator = creator;
                    task.difficulty_string = diffName;
                    task.display_name = `${artist} - ${title} (${creator}) [${diffName}]`;

                    QueueManager.add(task);
                    tasksAdded++;
                }
            });

            if (tasksAdded > 0 && typeof Utils !== 'undefined') {
                Utils.showFlashMessage(
                    `Added ${tasksAdded} task${tasksAdded > 1 ? 's' : ''} to queue`,
                    'success'
                );
            }
        };
    }

    // Start queue button (if we add one)
    const startQueueBtn = document.getElementById("startQueueBtn");
    if (startQueueBtn) {
        startQueueBtn.onclick = () => QueueManager.start();
    }

    // Initial render handled by app.js QueueUI.init()
});


/* ================================================================== */
/*  GLOBAL API (for app.js and other scripts)                         */
/* ================================================================== */

window.queueAPI = {
    hasJobs: () => QueueManager.hasPending(),
    isRunning: () => QueueManager.isRunning(),
    start: () => QueueManager.start(),
    clear: () => QueueManager.clear(),
    stop: () => QueueManager.stop(),
    getLength: () => QueueManager.getLength()
};

window.MapperManager = MapperManager;
window.QueueManager = QueueManager;
