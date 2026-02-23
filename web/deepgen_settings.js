import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "deepgen.Settings",

    async setup() {
        // Fetch the current API key on boot
        let currentApiKey = "";
        try {
            const resp = await api.fetchApi("/deepgen/get_api_key", { method: "GET" });
            if (resp.status === 200) {
                const data = await resp.json();
                currentApiKey = data.api_key || "";
            }
        } catch (e) {
            console.error("Failed to fetch DeepGen API Key", e);
        }

        // Add setting to the ComfyUI Settings menu
        app.ui.settings.addSetting({
            id: "DeepGen.api_key",
            name: "DeepGen API Key",
            type: "text",
            defaultValue: currentApiKey,
            async onChange(newVal) {
                if (newVal === currentApiKey) return;
                try {
                    const resp = await api.fetchApi("/deepgen/set_api_key", {
                        method: "POST",
                        body: JSON.stringify({ api_key: newVal }),
                        headers: { "Content-Type": "application/json" }
                    });
                    if (resp.status === 200) {
                        currentApiKey = newVal;
                        console.log("DeepGen API Key saved successfully.");
                    } else {
                        console.error("Failed to save DeepGen API Key");
                    }
                } catch (e) {
                    console.error("Error saving DeepGen API Key", e);
                }
            }
        });

        // Show warning if key is missing or is the default placeholder
        if (!currentApiKey || currentApiKey === "<your_deepgen_api_key_here>") {
            setTimeout(() => {
                alert("DeepGen Nodes: API Key is missing!\n\nPlease click the ComfyUI Settings gear icon and enter your DeepGen API Key to use the nodes.");
            }, 1000);
        }
    }
});
