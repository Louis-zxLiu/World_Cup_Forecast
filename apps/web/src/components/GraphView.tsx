import React, { useCallback } from "react";
import ReactFlow, {
  Background,
  Controls,
  Edge,
  Handle,
  Node,
  NodeProps,
  Position,
} from "reactflow";
import "reactflow/dist/style.css";

export type NodeStatus = "idle" | "running" | "done" | "error";

const NODE_LABELS: Record<string, string> = {
  supervisor: "统计预测",
  strength_node: "实力分析",
  form_node: "近期状态",
  news_node: "新闻舆情",
  odds_node: "赔率市场",
  debate_node: "多空辩论",
  risk_node: "风控评估",
  report_node: "生成报告",
};

const STATUS_COLORS: Record<NodeStatus, { bg: string; border: string; text: string }> = {
  idle:    { bg: "#f1f5f9", border: "#cbd5e1", text: "#64748b" },
  running: { bg: "#eff6ff", border: "#3b82f6", text: "#1d4ed8" },
  done:    { bg: "#f0fdf4", border: "#22c55e", text: "#15803d" },
  error:   { bg: "#fef2f2", border: "#ef4444", text: "#b91c1c" },
};

function AgentNode({ data }: NodeProps) {
  const status: NodeStatus = data.status ?? "idle";
  const col = STATUS_COLORS[status];
  return (
    <div
      style={{
        background: col.bg,
        border: `2px solid ${col.border}`,
        borderRadius: 8,
        padding: "6px 14px",
        fontSize: 12,
        fontWeight: 600,
        color: col.text,
        minWidth: 88,
        textAlign: "center",
        boxShadow: status === "running" ? `0 0 0 3px ${col.border}44` : undefined,
        transition: "all 0.25s ease",
      }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      {status === "running" && (
        <span style={{ marginRight: 4, display: "inline-block", animation: "spin 1s linear infinite" }}>⟳</span>
      )}
      {status === "done" && <span style={{ marginRight: 4 }}>✓</span>}
      {data.label}
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  );
}

const nodeTypes = { agent: AgentNode };

const STATIC_NODES: Node[] = [
  { id: "supervisor",    type: "agent", position: { x: 200, y: 0   }, data: { label: NODE_LABELS.supervisor    } },
  { id: "strength_node", type: "agent", position: { x: 0,   y: 100 }, data: { label: NODE_LABELS.strength_node } },
  { id: "form_node",     type: "agent", position: { x: 133, y: 100 }, data: { label: NODE_LABELS.form_node     } },
  { id: "news_node",     type: "agent", position: { x: 266, y: 100 }, data: { label: NODE_LABELS.news_node     } },
  { id: "odds_node",     type: "agent", position: { x: 399, y: 100 }, data: { label: NODE_LABELS.odds_node     } },
  { id: "debate_node",   type: "agent", position: { x: 200, y: 200 }, data: { label: NODE_LABELS.debate_node   } },
  { id: "risk_node",     type: "agent", position: { x: 200, y: 300 }, data: { label: NODE_LABELS.risk_node     } },
  { id: "report_node",   type: "agent", position: { x: 200, y: 400 }, data: { label: NODE_LABELS.report_node   } },
];

const EDGES: Edge[] = [
  { id: "sup-str",   source: "supervisor",    target: "strength_node", type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
  { id: "sup-frm",   source: "supervisor",    target: "form_node",     type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
  { id: "sup-nws",   source: "supervisor",    target: "news_node",     type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
  { id: "sup-odd",   source: "supervisor",    target: "odds_node",     type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
  { id: "str-deb",   source: "strength_node", target: "debate_node",   type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
  { id: "frm-deb",   source: "form_node",     target: "debate_node",   type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
  { id: "nws-deb",   source: "news_node",     target: "debate_node",   type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
  { id: "odd-deb",   source: "odds_node",     target: "debate_node",   type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
  { id: "deb-rsk",   source: "debate_node",   target: "risk_node",     type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
  { id: "rsk-rep",   source: "risk_node",     target: "report_node",   type: "smoothstep", animated: false, style: { stroke: "#94a3b8" } },
];

type Props = {
  nodeStates: Record<string, NodeStatus>;
};

export function GraphView({ nodeStates }: Props) {
  const nodes: Node[] = STATIC_NODES.map((n) => ({
    ...n,
    data: { ...n.data, status: nodeStates[n.id] ?? "idle" },
  }));

  const edges: Edge[] = EDGES.map((e) => {
    const srcDone = (nodeStates[e.source] ?? "idle") === "done";
    return {
      ...e,
      animated: srcDone,
      style: { stroke: srcDone ? "#22c55e" : "#94a3b8" },
    };
  });

  return (
    <div style={{ width: "100%", height: 520, borderRadius: 8, overflow: "hidden", background: "#f8fafc" }}>
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.3 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        preventScrolling={false}
      >
        <Background color="#e2e8f0" gap={16} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
