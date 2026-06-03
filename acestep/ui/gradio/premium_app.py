"""Premium ACE-Step Gradio shell with modular page imports."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import quote

import gradio as gr

from acestep.ui.gradio.events import (
    setup_event_handlers,
    setup_training_event_handlers,
)
from acestep.ui.gradio.events.generation.cancel_api import (
    CANCEL_GENERATION_ENDPOINT,
    register_generation_cancel_route,
)
from acestep.ui.gradio.events.wiring import (
    register_audio_processing_handlers,
    register_batch_folder_handlers,
    register_grid_testing_handlers,
    register_library_handlers,
    register_sam_audio_handlers,
    register_simple_create_handlers,
)
from acestep.ui.gradio.help_content import HELP_MODAL_CSS
from acestep.ui.gradio.i18n import get_i18n, t
from acestep.ui.gradio.interfaces.result import create_results_section
from acestep.ui.gradio.interfaces.audio_player_preferences import (
    get_audio_player_preferences_head,
)
from acestep.ui.gradio.interfaces.user_preferences import (
    get_user_preferences_head,
    wire_preference_restore,
)
from acestep.ui.gradio.pages import (
    create_audio_processing_page,
    create_batch_folder_page,
    create_dataset_page,
    create_generation_workspace_page,
    create_grid_testing_page,
    create_library_page,
    create_sam_audio_page,
    create_simple_create_page,
    create_studio_page,
    create_training_page,
)
from acestep.ui.gradio.premium_preset_components import (
    build_preset_component_map,
    preset_components_for_keys,
)
from acestep.ui.gradio.premium_preset_value_safety import (
    component_specs_from_components,
)
from acestep.ui.gradio.premium_features import (
    delete_preset_action,
    get_preset_component_keys,
    load_preset_action,
    open_models_folder,
    open_outputs_folder,
    refresh_dashboard,
    save_preset_action,
    startup_preset_updates,
)


APP_BROWSER_TITLE = "ACE-Step 1.5 XL Premium v4.1"
APP_RELEASE_URL = "https://www.patreon.com/posts/157675060"
APP_HEADER_MARKDOWN = f"# {APP_BROWSER_TITLE} : [{APP_RELEASE_URL}]({APP_RELEASE_URL})"
_FAVICON_PATH = Path(__file__).resolve().parent / "assets" / "ace_step_premium_favicon.svg"
_TOOLTIP_SCRIPT = """
<script>
document.addEventListener('mouseover', function(e) {
    var el = e.target.closest('.has-info-container, .checkbox-container');
    if (!el) return;
    var rect = el.getBoundingClientRect();
    if (rect.bottom > window.innerHeight * 0.65) {
        el.classList.add('tooltip-flip');
    } else {
        el.classList.remove('tooltip-flip');
    }
});
</script>
"""

_BUTTON_PERSONALIZATION_SCRIPT = """
<script>
(function() {
    const cancelEndpoint = "__ACE_CANCEL_ENDPOINT__";
    const colorPalette = [
        ["#1d4ed8", "#0e7490"],
        ["#6d28d9", "#7e22ce"],
        ["#be185d", "#9f1239"],
        ["#b91c1c", "#c2410c"],
        ["#92400e", "#854d0e"],
        ["#3f6212", "#166534"],
        ["#047857", "#0f766e"],
        ["#155e75", "#0369a1"],
        ["#3730a3", "#4338ca"],
        ["#581c87", "#86198f"],
        ["#831843", "#be123c"],
        ["#7f1d1d", "#b91c1c"],
        ["#7c2d12", "#ea580c"],
        ["#365314", "#4d7c0f"],
        ["#064e3b", "#047857"],
        ["#164e63", "#0891b2"],
        ["#312e81", "#4f46e5"],
        ["#4c1d95", "#7c3aed"],
        ["#701a75", "#a21caf"],
        ["#881337", "#e11d48"],
        ["#78350f", "#b45309"],
        ["#1e3a8a", "#2563eb"],
        ["#0f172a", "#334155"],
        ["#134e4a", "#0f766e"],
        ["#1e40af", "#1d4ed8"],
        ["#5b21b6", "#6d28d9"],
        ["#9d174d", "#db2777"],
        ["#991b1b", "#dc2626"],
        ["#713f12", "#a16207"],
        ["#14532d", "#15803d"],
        ["#0c4a6e", "#0284c7"],
        ["#4338ca", "#0f766e"],
        ["#7e22ce", "#be185d"],
        ["#9f1239", "#c2410c"],
        ["#0f766e", "#1d4ed8"],
        ["#047857", "#7c2d12"],
        ["#2563eb", "#9333ea"],
        ["#be123c", "#ea580c"],
        ["#166534", "#0e7490"],
        ["#334155", "#0f766e"],
    ];

    const emojiRules = [
        [/generate|music/i, "🎵"],
        [/analy[sz]e|scan|search|get item/i, "🔍"],
        [/convert|codes|restore|apply/i, "🔄"],
        [/transcribe|record/i, "🎙️"],
        [/caption|enhance|auto-fill/i, "✨"],
        [/lyric|lrc|timestamp/i, "🎼"],
        [/random|sample|click/i, "🎲"],
        [/load|import/i, "📥"],
        [/save|export/i, "💾"],
        [/delete|unload|clear|remove/i, "🗑️"],
        [/open|folder/i, "📁"],
        [/refresh|reset/i, "🔄"],
        [/initialize|service|settings/i, "⚙️"],
        [/preprocess|process/i, "⚡"],
        [/stop|cancel/i, "🛑"],
        [/train|start/i, "🚀"],
        [/score|quality/i, "📊"],
        [/remix|repaint/i, "🎛️"],
        [/previous/i, "◀️"],
        [/next/i, "▶️"],
    ];

    function visible(el) {
        const rect = el.getBoundingClientRect();
        const style = window.getComputedStyle(el);
        return rect.width > 2
            && rect.height > 2
            && style.display !== "none"
            && style.visibility !== "hidden";
    }

    function buttonText(button) {
        return (button.innerText || button.textContent || button.getAttribute("aria-label") || "")
            .trim()
            .replace(/\\s+/g, " ");
    }

    function isCommandButton(button) {
        if (!(button instanceof HTMLButtonElement)) return false;
        if (button.getAttribute("role") === "tab") return false;
        if (button.closest('[role="tablist"], .tab-nav')) return false;

        const className = String(button.className || "");
        if (
            className.includes("help-inline-btn")
            || className.includes("reset-button")
            || className.includes("icon-button")
            || className.includes("header-button")
            || className.includes("label-wrap")
            || className.includes("show-api")
            || className.includes("settings")
            || className.includes("disable_click")
        ) {
            return false;
        }

        const label = buttonText(button);
        if (!label || label === "?" || label === "↻") return false;
        if (label.includes("Drop Audio Here") || label.includes("Click to Upload")) return false;

        return className.includes("primary")
            || className.includes("secondary")
            || className.includes("stop")
            || className.includes("action-btn");
    }

    function selectedTabKey() {
        return Array.from(document.querySelectorAll('button[role="tab"][aria-selected="true"], button[role="tab"].selected'))
            .map((tab) => buttonText(tab))
            .join("|");
    }

    function hashText(text) {
        let hash = 0;
        for (let i = 0; i < text.length; i += 1) {
            hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
        }
        return Math.abs(hash);
    }

    function hasDecorativePrefix(label) {
        return /^[\\u{1F300}-\\u{1FAFF}\\u23E9-\\u23FA\\u2600-\\u27BF\\u2190-\\u21FF]/u.test(label);
    }

    function stripDecorativePrefix(label) {
        if (!hasDecorativePrefix(label)) return label;
        return label.replace(/^\\S+\\s+/, "").trim();
    }

    function usesDynamicServerLabel(button) {
        return Boolean(button.closest(".action-btn-generate"));
    }

    function emojiFor(label) {
        for (const [pattern, emoji] of emojiRules) {
            if (pattern.test(label)) return emoji;
        }
        return "◆";
    }

    function colorFor(tabKey, index) {
        const base = hashText(tabKey || "ACE-Step") % 360;
        const [from, to] = colorPalette[(base + index) % colorPalette.length];
        return {
            gradient: `linear-gradient(135deg, ${from} 0%, ${to} 100%)`,
            shadow: `0 8px 20px color-mix(in srgb, ${from} 28%, transparent)`,
        };
    }

    function cancelColorFor(button) {
        if (button.closest(".action-btn-cancel-simple")) {
            return {
                gradient: "linear-gradient(135deg, #991b1b 0%, #e11d48 100%)",
                shadow: "0 8px 20px rgba(225, 29, 72, 0.28)",
            };
        }
        if (button.closest(".action-btn-cancel-advanced")) {
            return {
                gradient: "linear-gradient(135deg, #7c2d12 0%, #f97316 100%)",
                shadow: "0 8px 20px rgba(249, 115, 22, 0.28)",
            };
        }
        if (button.closest(".action-btn-cancel-batch")) {
            return {
                gradient: "linear-gradient(135deg, #4c1d95 0%, #7e22ce 100%)",
                shadow: "0 8px 20px rgba(126, 34, 206, 0.28)",
            };
        }
        return null;
    }

    function setStyleProperty(button, name, value) {
        if (button.style.getPropertyValue(name) !== value) {
            button.style.setProperty(name, value);
        }
    }

    function isCancelButton(button) {
        return button instanceof HTMLButtonElement
            && Boolean(
                button.closest(".action-btn-cancel-simple")
                || button.closest(".action-btn-cancel-advanced")
                || button.closest(".action-btn-cancel-batch")
            );
    }

    function subprocessModeEnabled() {
        const container = document.getElementById("acestep-subprocess-mode-checkbox");
        const input = container && container.matches('input[type="checkbox"]')
            ? container
            : (
                container
                    ? container.querySelector('input[type="checkbox"]')
                    : document.querySelector('#acestep-subprocess-mode-checkbox input[type="checkbox"]')
        );
        return Boolean(input && input.checked);
    }

    function showTemporaryButtonText(button, message, restoreDelay) {
        const originalText = buttonText(button);
        button.textContent = message;
        window.setTimeout(() => {
            button.textContent = originalText;
            schedulePersonalization();
        }, restoreDelay);
    }

    async function requestCancel(button) {
        const isBatchCancel = Boolean(button.closest(".action-btn-cancel-batch"));
        const message = isBatchCancel
            ? "Are you sure you want to cancel the current generation and the remaining batch?"
            : "Are you sure you want to cancel the current generation?";
        if (!window.confirm(message)) return;

        const originalText = buttonText(button);
        if (!subprocessModeEnabled()) {
            showTemporaryButtonText(button, "Subprocess Mode Off", 1200);
            return;
        }
        button.textContent = "Cancelling...";
        try {
            const response = await fetch(cancelEndpoint, {
                method: "POST",
                headers: {"Accept": "application/json"},
            });
            if (!response.ok) {
                throw new Error(`Cancel request failed: HTTP ${response.status}`);
            }
            const payload = await response.json();
            button.textContent = payload.active ? "Subprocess Cancel Requested" : "No Active Subprocess";
            window.setTimeout(() => {
                button.textContent = originalText;
                schedulePersonalization();
            }, 1200);
        } catch (error) {
            console.error("[ACE-Step] Cancel request failed", error);
            button.textContent = "Cancel Failed";
            window.setTimeout(() => {
                button.textContent = originalText;
                schedulePersonalization();
            }, 1800);
        }
    }

    document.addEventListener("click", function(event) {
        const button = event.target instanceof Element ? event.target.closest("button") : null;
        if (!isCancelButton(button)) return;
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        requestCancel(button);
    }, true);

    function personalizeButtons() {
        const tabKey = selectedTabKey();
        const buttons = Array.from(document.querySelectorAll("button"))
            .filter(isCommandButton)
            .filter(visible);

        buttons.forEach((button, index) => {
            const color = cancelColorFor(button) || colorFor(tabKey, index);
            button.dataset.aceCommandButton = "true";
            button.dataset.aceButtonOrdinal = String(index + 1);
            setStyleProperty(button, "--ace-btn-bg", color.gradient);
            setStyleProperty(button, "--ace-btn-shadow", color.shadow);
        });
    }

    let scheduled = false;
    function schedulePersonalization() {
        if (scheduled) return;
        scheduled = true;
        window.requestAnimationFrame(() => {
            scheduled = false;
            personalizeButtons();
        });
    }

    document.addEventListener("click", schedulePersonalization, true);
    document.addEventListener("input", schedulePersonalization, true);
    document.addEventListener("change", schedulePersonalization, true);
    new MutationObserver(schedulePersonalization).observe(document.documentElement, {
        childList: true,
        subtree: true,
    });

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", schedulePersonalization, { once: true });
    } else {
        schedulePersonalization();
    }
    window.setTimeout(schedulePersonalization, 400);
    window.setTimeout(schedulePersonalization, 1400);
    window.setTimeout(schedulePersonalization, 3000);
})();
</script>
""".replace("__ACE_CANCEL_ENDPOINT__", CANCEL_GENERATION_ENDPOINT)

_UNAVAILABLE_GENERATION_MODE_SCRIPT = """
<script>
(function() {
    const MODE_SELECTOR = "#acestep-generation-mode";

    function updateModeAvailability() {
        const root = document.querySelector(MODE_SELECTOR);
        if (!root) return;

        root.querySelectorAll('label[data-testid$="-radio-label"]').forEach((label) => {
            const testId = label.getAttribute("data-testid") || "";
            const unavailable = testId.includes("unavailable");
            const input = label.querySelector('input[type="radio"]');
            if (!input) return;

            if (unavailable) {
                input.dataset.aceModeDisabled = "true";
                if (!input.disabled) input.disabled = true;
                label.classList.add("ace-mode-unavailable");
                label.setAttribute("aria-disabled", "true");
                return;
            }

            if (input.dataset.aceModeDisabled === "true") {
                delete input.dataset.aceModeDisabled;
                input.disabled = false;
                label.classList.remove("ace-mode-unavailable");
                label.removeAttribute("aria-disabled");
            }
        });
    }

    let scheduled = false;
    function scheduleModeAvailability() {
        if (scheduled) return;
        scheduled = true;
        window.requestAnimationFrame(() => {
            scheduled = false;
            updateModeAvailability();
        });
    }

    document.addEventListener("change", scheduleModeAvailability, true);
    new MutationObserver(scheduleModeAvailability).observe(document.documentElement, {
        attributeFilter: ["data-testid"],
        attributes: true,
        childList: true,
        subtree: true,
    });

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", scheduleModeAvailability, { once: true });
    } else {
        scheduleModeAvailability();
    }
    window.setTimeout(scheduleModeAvailability, 400);
    window.setTimeout(scheduleModeAvailability, 1400);
})();
</script>
"""

_PREMIUM_CSS = """
#acestep-generation-mode .ace-mode-unavailable {
    cursor: not-allowed !important;
    opacity: 0.52 !important;
    pointer-events: none !important;
}
  .ace-status-scroll-10,
  .ace-status-markdown {
      max-height: calc(1.45em * 10 + 1rem) !important;
      overflow-y: auto !important;
  }
  .ace-video-preview,
  .ace-video-preview .wrap,
  .ace-video-preview .wrap > video,
  .ace-video-preview video {
      max-height: 400px !important;
  }
  .ace-video-preview video {
      width: 100% !important;
      height: auto !important;
      object-fit: contain !important;
  }
  .gradio-container button[data-ace-command-button="true"] {
    background: var(--ace-btn-bg) !important;
    box-shadow: var(--ace-btn-shadow) !important;
    color: #ffffff !important;
    border: 0 !important;
    text-shadow: 0 1px 1px rgba(15, 23, 42, 0.34);
    font-size: 15px !important;
    font-weight: 800 !important;
    line-height: 1.18 !important;
    min-height: 42px !important;
    padding: 0.62rem 0.92rem !important;
    white-space: normal !important;
    overflow-wrap: anywhere;
    text-align: center !important;
    transition: transform 0.12s ease, box-shadow 0.16s ease, filter 0.16s ease;
}
.gradio-container button.lg[data-ace-command-button="true"] {
    font-size: 16px !important;
    min-height: 42px !important;
    padding: 0.62rem 0.92rem !important;
}
.gradio-container button.sm[data-ace-command-button="true"] {
    font-size: 15px !important;
    min-height: 42px !important;
    padding: 0.62rem 0.92rem !important;
}
.gradio-container button[data-ace-command-button="true"]:hover {
    transform: translateY(-1px);
    filter: saturate(1.08) brightness(1.02);
}
.gradio-container button[data-ace-command-button="true"]:active {
    transform: translateY(0);
}
.gradio-container button[data-ace-command-button="true"]:disabled {
    opacity: 0.68;
    transform: none;
}
.action-btn button,
button.action-btn {
    border: 0 !important;
    color: #ffffff !important;
    text-shadow: 0 1px 1px rgba(15, 23, 42, 0.32);
    font-weight: 800 !important;
    line-height: 1.18 !important;
    min-height: 42px !important;
    padding: 0.62rem 0.92rem !important;
    transition: transform 0.12s ease, box-shadow 0.16s ease, filter 0.16s ease;
}
.action-btn button *,
button.action-btn * {
    color: inherit !important;
}
.action-btn button:focus-visible,
button.action-btn:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.35), 0 8px 20px rgba(2, 6, 23, 0.18) !important;
}
.action-btn button:hover,
button.action-btn:hover {
    transform: translateY(-1px);
    filter: saturate(1.08) brightness(1.02);
}
.action-btn button:active,
button.action-btn:active {
    transform: translateY(0);
}
.action-btn button:disabled,
button.action-btn:disabled {
    opacity: 0.68;
    transform: none;
}
.gradio-container .action-btn,
.gradio-container button.action-btn,
.gradio-container button[data-ace-command-button="true"] {
    min-height: 42px !important;
    height: 46px !important;
    max-height: 46px !important;
    overflow: hidden !important;
}
.gradio-container .action-btn > div,
.gradio-container .action-btn button {
    min-height: 42px !important;
    height: 46px !important;
    max-height: 46px !important;
    overflow: hidden !important;
}
.ace-generate-action-row,
.ace-generate-action-row > div {
    min-height: 0 !important;
    height: auto !important;
    max-height: 56px !important;
    align-items: stretch !important;
    overflow: hidden !important;
}
.ace-generate-action-row .action-btn,
.ace-generate-action-row button {
    min-height: 42px !important;
    height: 46px !important;
    max-height: 46px !important;
    overflow: hidden !important;
}
.action-btn-upscale button,
button.action-btn-upscale {
    background: linear-gradient(135deg, #2563eb 0%, #0891b2 50%, #059669 100%) !important;
    box-shadow: 0 8px 20px rgba(8, 145, 178, 0.25) !important;
}
.action-btn-generate button,
button.action-btn-generate {
    background: linear-gradient(135deg, #dc2626 0%, #f97316 52%, #ec4899 100%) !important;
    box-shadow: 0 10px 24px rgba(220, 38, 38, 0.28) !important;
}
.action-btn-generate-song button,
button.action-btn-generate-song {
    background: linear-gradient(
        180deg,
        #d2b1fe 0%,
        #6d28d9 52%,
        #2e1065 100%
    ) !important;
    border: 1px solid #ddb8fe !important;
    box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.08),
        0 4px 0 #260d53,
        0 14px 22px rgba(109, 40, 217, 0.24) !important;
    isolation: isolate;
    overflow: hidden !important;
    position: relative !important;
    text-shadow: 0 2px 0 rgba(0, 0, 0, 0.55),
        0 0 9px rgba(255, 255, 255, 0.18);
}
.action-btn-generate-song button::before,
button.action-btn-generate-song::before {
    background: linear-gradient(
        90deg,
        transparent,
        rgba(255, 255, 255, 0.2),
        transparent
    );
    border-radius: inherit;
    content: "";
    inset: 0 -20%;
    pointer-events: none;
    position: absolute;
}
.action-btn-upscale button:hover,
button.action-btn-upscale:hover {
    box-shadow: 0 10px 24px rgba(8, 145, 178, 0.34) !important;
}
.action-btn-generate button:hover,
button.action-btn-generate:hover {
    box-shadow: 0 12px 28px rgba(220, 38, 38, 0.36) !important;
}
.action-btn-generate-song button:hover,
button.action-btn-generate-song:hover {
    box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.1),
        0 5px 0 #260d53,
        0 16px 26px rgba(109, 40, 217, 0.3) !important;
}
.action-btn-preview button,
button.action-btn-preview {
    background: linear-gradient(135deg, #2563eb 0%, #0e7490 100%) !important;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.24) !important;
}
.action-btn-preview button:hover,
button.action-btn-preview:hover {
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.32) !important;
}
.action-btn-cancel button,
button.action-btn-cancel {
    background: linear-gradient(135deg, #dc2626 0%, #f43f5e 100%) !important;
    box-shadow: 0 8px 20px rgba(220, 38, 38, 0.24) !important;
}
.action-btn-cancel button:hover,
button.action-btn-cancel:hover {
    box-shadow: 0 10px 24px rgba(220, 38, 38, 0.32) !important;
}
.action-btn-delete-preset button,
button.action-btn-delete-preset {
    background: linear-gradient(135deg, #dc2626 0%, #f43f5e 100%) !important;
    box-shadow: 0 8px 20px rgba(220, 38, 38, 0.24) !important;
}
.action-btn-delete-preset button:hover,
button.action-btn-delete-preset:hover {
    box-shadow: 0 10px 24px rgba(220, 38, 38, 0.32) !important;
}
.action-btn-cancel-simple button,
button.action-btn-cancel-simple {
    background: linear-gradient(135deg, #991b1b 0%, #e11d48 100%) !important;
    box-shadow: 0 8px 20px rgba(225, 29, 72, 0.28) !important;
}
.action-btn-cancel-advanced button,
button.action-btn-cancel-advanced {
    background: linear-gradient(135deg, #7c2d12 0%, #f97316 100%) !important;
    box-shadow: 0 8px 20px rgba(249, 115, 22, 0.28) !important;
}
.action-btn-cancel-batch button,
button.action-btn-cancel-batch {
    background: linear-gradient(135deg, #4c1d95 0%, #7e22ce 100%) !important;
    box-shadow: 0 8px 20px rgba(126, 34, 206, 0.28) !important;
}
.action-btn-open button,
button.action-btn-open {
    background: linear-gradient(135deg, #059669 0%, #14b8a6 100%) !important;
    box-shadow: 0 8px 20px rgba(5, 150, 105, 0.24) !important;
}
.action-btn-open button:hover,
button.action-btn-open:hover {
    box-shadow: 0 10px 24px rgba(5, 150, 105, 0.32) !important;
}
.action-btn-clear button,
button.action-btn-clear {
    background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
    box-shadow: 0 8px 20px rgba(234, 88, 12, 0.24) !important;
}
.action-btn-clear button:hover,
button.action-btn-clear:hover {
    box-shadow: 0 10px 24px rgba(234, 88, 12, 0.32) !important;
}
""" + HELP_MODAL_CSS


def _build_head(service_mode: bool) -> str:
    """Return the HTML head snippet used by the premium shell."""

    favicon_href = _load_favicon_data_uri()
    branding_head = _build_branding_head(
        title=APP_BROWSER_TITLE,
        favicon_href=favicon_href,
    )
    return (
        branding_head
        + get_audio_player_preferences_head()
        + ("" if service_mode else get_user_preferences_head())
        + _TOOLTIP_SCRIPT
        + _BUTTON_PERSONALIZATION_SCRIPT
        + _UNAVAILABLE_GENERATION_MODE_SCRIPT
    )


@lru_cache(maxsize=1)
def _load_favicon_data_uri() -> str:
    """Return the bundled premium favicon as an SVG data URI."""

    svg = _FAVICON_PATH.read_text(encoding="utf-8")
    return f"data:image/svg+xml,{quote(svg)}"


def _build_branding_head(*, title: str, favicon_href: str) -> str:
    """Return branding tags/scripts for browser-tab title and favicon."""

    title_js = json.dumps(title)
    favicon_js = json.dumps(favicon_href)
    return f"""
<link rel="icon" type="image/svg+xml" href="{favicon_href}">
<link rel="shortcut icon" type="image/svg+xml" href="{favicon_href}">
<meta name="apple-mobile-web-app-title" content="{title}">
<script>
(function() {{
    const appTitle = {title_js};
    const faviconHref = {favicon_js};

    function ensureTitle() {{
        if (document.title !== appTitle) {{
            document.title = appTitle;
        }}
    }}

    function ensureFavicon(relValue) {{
        let link = document.querySelector('link[rel=\"' + relValue + '\"]');
        if (!link) {{
            link = document.createElement('link');
            link.setAttribute('rel', relValue);
            document.head.appendChild(link);
        }}
        link.setAttribute('type', 'image/svg+xml');
        link.setAttribute('href', faviconHref);
    }}

    function applyBranding() {{
        ensureTitle();
        ensureFavicon('icon');
        ensureFavicon('shortcut icon');
    }}

    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', applyBranding, {{ once: true }});
    }} else {{
        applyBranding();
    }}

    window.addEventListener('focus', applyBranding);
    window.setTimeout(applyBranding, 150);
    window.setTimeout(applyBranding, 1200);
}})();
</script>
"""


def create_gradio_interface(
    dit_handler: Any,
    llm_handler: Any,
    dataset_handler: Any,
    init_params: dict[str, Any] | None = None,
    language: str = "en",
) -> gr.Blocks:
    """Create the premium ACE-Step Gradio interface."""

    _ = get_i18n(language)
    service_mode = init_params is not None and init_params.get("service_mode", False)
    launch_theme = gr.themes.Default()
    launch_head = _build_head(service_mode)

    with gr.Blocks(
        title=APP_BROWSER_TITLE,
    ) as demo:
        gr.Markdown(APP_HEADER_MARKDOWN)

        with gr.Tabs():
            with gr.Tab("Generate Song", render_children=True):
                simple_page = create_simple_create_page(init_params=init_params)

            with gr.Tab("Advanced", render_children=True):
                create_page = create_generation_workspace_page(
                    dit_handler=dit_handler,
                    llm_handler=llm_handler,
                    init_params=init_params,
                    language=language,
                    include_results=False,
                )

            with gr.Tab("Audio Processing", render_children=True):
                audio_processing_page = create_audio_processing_page()

            with gr.Tab("SAM Audio Segment", render_children=True):
                sam_audio_page = create_sam_audio_page()

            with gr.Tab("Library", render_children=True):
                library_page = create_library_page()

            with gr.Tab("Results", render_children=True):
                with gr.Column(visible=True) as results_wrapper:
                    results_section = create_results_section(dit_handler)

            with gr.Tab("Custom Preset System", render_children=True):
                studio_page = create_studio_page()

            with gr.Tab("Dataset", render_children=True):
                dataset_section = create_dataset_page(dataset_handler)

            with gr.Tab(t("training.tab_title"), visible=not service_mode, render_children=True):
                training_section = create_training_page(
                    dit_handler=dit_handler,
                    llm_handler=llm_handler,
                    init_params=init_params,
                )

            with gr.Tab("Batch Folder Processing", visible=not service_mode, render_children=True):
                batch_folder_section = create_batch_folder_page()

            with gr.Tab("Grid Testing", visible=not service_mode, render_children=True):
                grid_section = create_grid_testing_page()

        generation_section: dict[str, Any] = {}
        generation_section.update(create_page["settings_section"])
        generation_section.update(create_page["generation_section"])
        generation_section.update(audio_processing_page)
        generation_section.update(sam_audio_page)
        generation_section["results_wrapper"] = results_wrapper
        generation_section["subprocess_mode_checkbox"] = create_page[
            "subprocess_mode_checkbox"
        ]
        generation_section["simple_model_dropdown"] = simple_page[
            "simple_model_dropdown"
        ]
        generation_section["simple_lora_dropdown"] = simple_page[
            "simple_lora_dropdown"
        ]
        generation_section["simple_lora_scale_slider"] = simple_page[
            "simple_lora_scale_slider"
        ]
        generation_section["simple_quantization"] = simple_page["simple_quantization"]
        for key in (
            "dataset_model_config",
            "dataset_vram_preset",
            "dataset_name",
            "all_instrumental",
            "format_lyrics",
            "transcribe_lyrics",
            "lm_lyrics_language",
            "custom_tag",
            "use_only_custom_trigger",
            "tag_position",
            "genre_ratio",
            "skip_metas",
            "only_unlabeled",
            "auto_label_output_dir",
            "auto_label_subprocess",
            "auto_label_batch_size",
        ):
            generation_section[key] = training_section[key]

        setup_event_handlers(
            demo=demo,
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            dataset_handler=dataset_handler,
            dataset_section=dataset_section,
            generation_section=generation_section,
            results_section=results_section,
        )

        setup_training_event_handlers(
            demo=demo,
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            training_section=training_section,
            generation_section=generation_section,
        )
        register_batch_folder_handlers(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            batch_section=batch_folder_section,
            generation_section=generation_section,
            results_section=results_section,
        )
        register_grid_testing_handlers(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            grid_section=grid_section,
            simple_page=simple_page,
            generation_section=generation_section,
            results_section=results_section,
        )

        register_simple_create_handlers(
            simple_page=simple_page,
            generation_section=generation_section,
            results_section=results_section,
            dit_handler=dit_handler,
            llm_handler=llm_handler,
        )
        register_audio_processing_handlers(audio_processing_page)
        register_sam_audio_handlers(
            sam_audio_page,
            dit_handler=dit_handler,
            llm_handler=llm_handler,
        )
        register_library_handlers(library_page, demo=demo)

        preset_keys = get_preset_component_keys()
        preset_component_map = build_preset_component_map(
            generation_section=generation_section,
            simple_page=simple_page,
            training_section=training_section,
            dataset_section=dataset_section,
            batch_folder_section=batch_folder_section,
            audio_processing_section=audio_processing_page,
            sam_audio_section=sam_audio_page,
        )
        preset_components = preset_components_for_keys(preset_component_map, preset_keys)
        preset_component_specs = component_specs_from_components(
            preset_keys,
            preset_components,
        )
        preset_default_values = gr.State(
            [getattr(component, "value", None) for component in preset_components]
        )

        def startup_preset_updates_with_specs() -> tuple[Any, ...]:
            """Load startup preset values sanitized for this Gradio app."""

            return startup_preset_updates(preset_component_specs)

        def load_preset_action_with_specs(preset_name: str | None) -> tuple[Any, ...]:
            """Load preset values sanitized for this Gradio app."""

            return load_preset_action(preset_name, preset_component_specs)

        def delete_preset_action_with_specs(
            preset_name: str | None,
            default_values: list[Any] | tuple[Any, ...] | None = None,
            preset_name_input: str | None = None,
        ) -> tuple[Any, ...]:
            """Delete a preset and sanitize replacement values for this Gradio app."""

            return delete_preset_action(
                preset_name,
                default_values,
                preset_name_input,
                preset_component_specs,
            )

        startup_preset_updates_with_specs.__name__ = "startup_preset_updates"
        load_preset_action_with_specs.__name__ = "load_preset_action"
        delete_preset_action_with_specs.__name__ = "delete_preset_action"

        demo.load(
            fn=startup_preset_updates_with_specs,
            outputs=preset_components
            + [
                generation_section["lora_status"],
                generation_section["use_lora_checkbox"],
                studio_page["preset_dropdown"],
                studio_page["preset_status"],
                studio_page["studio_overview"],
            ],
        )
        preset_load_outputs = preset_components + [
            generation_section["lora_status"],
            generation_section["use_lora_checkbox"],
            studio_page["preset_dropdown"],
            studio_page["preset_status"],
            studio_page["studio_overview"],
        ]
        studio_page["preset_dropdown"].change(
            fn=load_preset_action_with_specs,
            inputs=[studio_page["preset_dropdown"]],
            outputs=preset_load_outputs,
        )
        studio_page["load_preset_btn"].click(
            fn=load_preset_action_with_specs,
            inputs=[studio_page["preset_dropdown"]],
            outputs=preset_load_outputs,
        )
        studio_page["save_preset_btn"].click(
            fn=save_preset_action,
            inputs=[
                studio_page["preset_name_input"],
                studio_page["preset_dropdown"],
                *preset_components,
            ],
            outputs=[
                studio_page["preset_dropdown"],
                studio_page["preset_status"],
                studio_page["studio_overview"],
            ],
        )
        studio_page["delete_preset_btn"].click(
            fn=delete_preset_action_with_specs,
            inputs=[
                studio_page["preset_dropdown"],
                preset_default_values,
                studio_page["preset_name_input"],
            ],
            outputs=preset_load_outputs,
        )
        studio_page["refresh_dashboard_btn"].click(
            fn=refresh_dashboard,
            inputs=[
                studio_page["preset_dropdown"],
                create_page["subprocess_mode_checkbox"],
            ],
            outputs=[studio_page["studio_overview"]],
        )
        create_page["subprocess_mode_checkbox"].change(
            fn=refresh_dashboard,
            inputs=[
                studio_page["preset_dropdown"],
                create_page["subprocess_mode_checkbox"],
            ],
            outputs=[studio_page["studio_overview"]],
        )
        studio_page["open_outputs_btn"].click(
            fn=open_outputs_folder,
            outputs=[studio_page["studio_status"]],
        )
        studio_page["open_models_btn"].click(
            fn=open_models_folder,
            outputs=[studio_page["studio_status"]],
        )

        wire_preference_restore(
            demo,
            generation_section,
            service_mode=service_mode,
        )

    register_generation_cancel_route(demo)
    demo._ace_launch_theme = launch_theme
    demo._ace_launch_css = _PREMIUM_CSS
    demo._ace_launch_head = launch_head
    return demo
