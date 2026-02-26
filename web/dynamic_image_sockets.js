import { app } from "../../scripts/app.js";

// Keep a cached configuration from the backend
let modelsConfig = null;

async function fetchModelsConfig() {
    if (modelsConfig) return modelsConfig;
    try {
        const response = await fetch("/deepgen/models");
        const data = await response.json();
        modelsConfig = data.models || [];
        return modelsConfig;
    } catch (e) {
        console.error("DeepGen: Event listener failed to fetch model configs.", e);
        return [];
    }
}

app.registerExtension({
    name: "DeepGen.DynamicImageSockets",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Image_deepgen") {
            // Intercept node configure to rebuild saved JSON inputs synchronously
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                if (info && info.inputs) {
                    // Synchronously add image sockets saved in the JSON so ComfyUI links match
                    for (const input of info.inputs) {
                        if (input.name.startsWith("image_") && input.type === "IMAGE") {
                            const exists = this.inputs && this.inputs.find(inp => inp.name === input.name);
                            if (!exists) {
                                this.addInput(input.name, input.type);
                            }
                        }
                    }
                }
                if (onConfigure) {
                    return onConfigure.apply(this, arguments);
                }
            };

            // Intercept node creation
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const updateSockets = async (node) => {
                    const configs = await fetchModelsConfig();

                    // Find the model widget
                    const modelWidget = node.widgets?.find(w => w.name === "model");
                    if (!modelWidget || !configs.length) return;

                    const selectedModelName = modelWidget.value;
                    const modelConfig = configs.find(c => c.name === selectedModelName);
                    const targetImages = modelConfig ? modelConfig.nb_of_images : 1;

                    if (!node.inputs) return;

                    // 1. Remove any image_X sockets that are beyond the target amount
                    // We iterate backwards to safely remove inputs
                    for (let i = node.inputs.length - 1; i >= 0; i--) {
                        if (node.inputs[i].name.startsWith("image_")) {
                            const match = node.inputs[i].name.match(/image_(\d+)/);
                            if (match) {
                                const idx = parseInt(match[1]);
                                if (idx > targetImages) {
                                    node.removeInput(i);
                                }
                            }
                        }
                    }

                    // 2. Add any missing image_X sockets up to target amount
                    for (let i = 1; i <= targetImages; i++) {
                        const socketName = `image_${i}`;
                        const exists = node.inputs.find(inp => inp.name === socketName);
                        if (!exists) {
                            node.addInput(socketName, "IMAGE");
                        }
                    }

                    // Force a UI redraw
                    if (app.graph) {
                        app.graph.setDirtyCanvas(true, true);
                    }
                };

                // Add properties to the widget to detect change
                setTimeout(() => {
                    const modelWidget = this.widgets?.find(w => w.name === "model");
                    if (modelWidget) {
                        // Hook the callback when the widget value changes
                        const callback = modelWidget.callback;
                        modelWidget.callback = (value) => {
                            if (callback) {
                                callback.call(modelWidget, value);
                            }
                            updateSockets(this);
                        };

                        // Initial update based on default value
                        updateSockets(this);
                    }
                }, 100);

                return r;
            };
        }
    }
});
