import { app } from "/scripts/app.js";

const NODE_NAME = "OpenAICompatibleImageGenerate";
const RATIOS = {
  "1:1 方形": [1, 1], "3:2 横向照片": [3, 2], "2:3 纵向照片": [2, 3],
  "4:3 横向": [4, 3], "3:4 纵向": [3, 4], "16:9 横屏": [16, 9],
  "9:16 竖屏": [9, 16], "5:4 横向": [5, 4], "4:5 竖向": [4, 5], "21:9 超宽屏": [21, 9]
};
const byName = (node, name) => node.widgets?.find(w => w.name === name);

function calc(node) {
  const mode = byName(node, "resolution_mode")?.value || "预设档位与比例";
  let w, h, ratioLabel, tierLabel;
  if (mode === "手动自定义尺寸") {
    w = Number(byName(node, "custom_width")?.value || 1024);
    h = Number(byName(node, "custom_height")?.value || 1024);
    const gcd = (a,b) => b ? gcd(b,a%b) : a;
    const d = gcd(Math.round(w), Math.round(h)) || 1;
    ratioLabel = `${Math.round(w/d)}:${Math.round(h/d)}`;
    tierLabel = "自定义尺寸";
  } else {
    const tier = byName(node, "resolution_tier")?.value || "1K";
    const long = ({"1K":1024,"2K":2048,"4K":4096})[tier] || 1024;
    const selected = byName(node, "aspect_ratio")?.value || "1:1 方形";
    const [rw,rh] = RATIOS[selected] || [1,1];
    if (rw >= rh) { w = long; h = Math.max(8, Math.round((long*rh/rw)/8)*8); }
    else { h = long; w = Math.max(8, Math.round((long*rw/rh)/8)*8); }
    ratioLabel = `${rw}:${rh}`;
    tierLabel = tier;
  }
  return {w,h,ratioLabel,tierLabel,mode};
}

function refresh(node) {
  const el = node.__resolutionPreview;
  if (!el) return;
  const {w,h,ratioLabel,tierLabel,mode} = calc(node);
  const maxW = 190, maxH = 118;
  const scale = Math.min(maxW / Math.max(w,1), maxH / Math.max(h,1));
  const rw = Math.max(24, w*scale), rh = Math.max(24, h*scale);
  el.querySelector(".oib-shape").style.width = `${rw}px`;
  el.querySelector(".oib-shape").style.height = `${rh}px`;
  el.querySelector(".oib-ratio").textContent = `画幅比例  ${ratioLabel}`;
  el.querySelector(".oib-dims").textContent = `${Math.round(w)} × ${Math.round(h)} px`;
  el.querySelector(".oib-meta").textContent = `${tierLabel} · ${(w*h/1e6).toFixed(2)} MP`;
  el.querySelector(".oib-mode").textContent = mode === "手动自定义尺寸" ? "手动输入" : "比例预设";
}

app.registerExtension({
  name: "OpenAIImageBridge.ResolutionPreview",
  nodeCreated(node) {
    if (node.comfyClass !== NODE_NAME) return;
    const wrap = document.createElement("div");
    wrap.className = "oib-resolution-preview";
    wrap.style.cssText = "box-sizing:border-box;width:100%;padding:10px 12px;margin:4px 0 8px;border:1px solid var(--border-color,#555);border-radius:9px;background:var(--comfy-input-bg,#222);color:var(--input-text,var(--fg-color,#eee));font:12px sans-serif;";
    wrap.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;opacity:.8"><b>分辨率实时预览</b><span class="oib-mode"></span></div><div style="height:126px;display:flex;align-items:center;justify-content:center;background:repeating-conic-gradient(#8882 0% 25%,transparent 0% 50%) 50%/16px 16px;border-radius:6px"><div class="oib-shape" style="max-width:190px;max-height:118px;border:2px solid #42a5f5;background:linear-gradient(135deg,#42a5f544,#42a5f511);border-radius:3px;box-sizing:border-box;transition:width .12s,height .12s"></div></div><div style="display:flex;justify-content:space-between;gap:8px;margin-top:9px;align-items:baseline"><b class="oib-ratio" style="font-size:13px">画幅比例 1:1</b><b class="oib-dims" style="font-size:14px;font-variant-numeric:tabular-nums">1024 × 1024 px</b></div><div class="oib-meta" style="margin-top:4px;opacity:.7">1K · 1.05 MP</div>`;
    node.__resolutionPreview = wrap;
    if (typeof node.addDOMWidget === "function") {
      node.addDOMWidget("resolution_preview", "div", wrap, {serialize:false, hideOnZoom:false});
    }
    for (const name of ["resolution_mode","resolution_tier","aspect_ratio","custom_width","custom_height"]) {
      const widget = byName(node,name);
      if (!widget) continue;
      const old = widget.callback;
      widget.callback = function(value) {
        if (old) old.call(this,value);
        refresh(node);
      };
    }
    refresh(node);
    node.setSize?.([node.size[0], Math.max(node.size[1], 520)]);
  }
});
