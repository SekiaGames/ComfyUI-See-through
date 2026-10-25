/**
 * SeeThrough 前端小工具
 *
 * 只干一件事：把 SavePSD 节点的 filename_prefix 跟着 LoadImage 的文件名自动填，
 * 保证「站姿图名 = 素材目录名 = 输出前缀」。
 *
 * PSD 现在是节点在服务端直接写出来的（psd-tools），不再需要浏览器里的
 * 「Download PSD」按钮，所以这个文件里没有 ag-psd 相关的代码了。
 */

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const PLACEHOLDER_PREFIXES = new Set(["", "image", "seethrough", "rqx"]);

function imagePrefix(filename) {
    if (!filename) return "";
    let name = String(filename);
    name = name.replace(/^\(pid-[^)]*\)/, "");       // 粘贴的图会带 (pid-1234) 前缀
    name = name.split(/[\\/]/).pop();                 // 去掉子目录
    name = name.replace(/\.[^.]+$/, "");              // 去扩展名
    return name.replace(/[^\p{L}\p{N}_-]+/gu, "_").replace(/^_+|_+$/g, "");
}

function findPrefixWidget(node) {
    return (node.widgets || []).find((w) => w.name === "filename_prefix");
}

function syncPrefix(graph) {
    if (!graph || !graph._nodes) return;
    const saves = graph._nodes.filter((n) => n.comfyClass === "SeeThrough_SavePSD");
    if (!saves.length) return;
    const loads = graph._nodes.filter((n) => n.comfyClass === "LoadImage");
    if (!loads.length) return;
    const stem = imagePrefix(loads[0].widgets?.[0]?.value);
    if (!stem) return;

    for (const s of saves) {
        const w = findPrefixWidget(s);
        if (!w) continue;
        // 用户手动改过就不动它；只覆盖占位值或上次自动填的值
        const auto = w._autoFrom;
        if (w.value && !PLACEHOLDER_PREFIXES.has(String(w.value).toLowerCase()) && w.value !== auto) continue;
        if (w.value === stem) { w._autoFrom = stem; continue; }
        w.value = stem;
        w._autoFrom = stem;
        s.setDirtyCanvas?.(true, true);
        console.log(`[SeeThrough] filename_prefix -> ${stem}（跟着输入图 ${loads[0].widgets?.[0]?.value}）`);
    }
}

app.registerExtension({
    name: "ComfyUI-See-through.Prefix",

    async nodeCreated(node) {
        if (node.comfyClass === "LoadImage" || node.comfyClass === "SeeThrough_SavePSD") {
            setTimeout(() => syncPrefix(node.graph), 0);
        }
        if (node.comfyClass === "LoadImage") {
            const orig = node.onWidgetChanged;
            node.onWidgetChanged = function (name, ...rest) {
                const r = orig?.apply(this, arguments);
                if (name === "image") setTimeout(() => syncPrefix(this.graph), 0);
                return r;
            };
        }
    },

    // 打开已有工作流时，节点是先建后灌值的，nodeCreated 里读到的还是空 widget
    loadedGraphNode(node) {
        if (node.comfyClass === "LoadImage") setTimeout(() => syncPrefix(node.graph), 0);
    },
});

console.log("[SeeThrough] prefix extension loaded");
