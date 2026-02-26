import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "DeepGen.DisplayNode",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Display_deepgen") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (this.widgets) {
                    const pos = this.widgets.findIndex((w) => w.name === "text_display");
                    if (pos !== -1) {
                        for (let i = pos; i < this.widgets.length; i++) {
                            this.widgets[i].onRemove?.();
                        }
                        this.widgets.length = pos;
                    }
                }

                if (message && message.text) {
                    const textWidget = ComfyWidgets["STRING"](this, "text_display", ["STRING", { multiline: true }], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.opacity = 0.6;
                    textWidget.value = message.text.join("");
                }
            };
        }
    }
});
