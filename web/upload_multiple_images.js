import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "deepgen.UploadMultipleImages",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "UploadMultipleImages_deepgen") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Add an explicit upload button widget
                this.addWidget("button", "Upload Images", "Upload Images", () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.multiple = true;
                    input.accept = "image/png,image/jpeg,image/webp,image/bmp";
                    input.onchange = async (e) => {
                        if (e.target.files.length === 0) return;
                        const uploadedNames = [];

                        // Find the hidden or visible text widget for image_paths
                        const imagePathsWidget = this.widgets.find((w) => w.name === "image_paths");
                        if (imagePathsWidget) {
                            imagePathsWidget.value = "Uploading...";
                        }

                        for (const file of e.target.files) {
                            try {
                                const body = new FormData();
                                body.append("image", file);
                                body.append("type", "input");

                                const resp = await api.fetchApi("/upload/image", {
                                    method: "POST",
                                    body,
                                });

                                if (resp.status === 200) {
                                    const data = await resp.json();
                                    uploadedNames.push(data.name);
                                } else {
                                    console.error(`Failed to upload ${file.name}`);
                                }
                            } catch (error) {
                                console.error(`Error uploading ${file.name}:`, error);
                            }
                        }

                        if (imagePathsWidget) {
                            // Update the widget value with the list of JSON-encoded filenames
                            imagePathsWidget.value = JSON.stringify(uploadedNames, null, 2);
                        }
                    };
                    input.click();
                });

                return r;
            };
        }
    },
});
