
import { useState } from "react";
import "./App.css";
const API_URL = import.meta.env.VITE_API_URL;

function App() {
  const [goal, setGoal] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const runAgent = async () => {
    if (!goal.trim()) {
      setError("Please enter a goal first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await (`${API_URL}/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Something went wrong.");
      }

      setResult(data.evaluation);
    } catch (err) {
      setError(err.message || "Could not connect to backend.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="app">
      <header className="header">
        <div className="logo">AI</div>
        <div>
          <h1>Agent Evaluation Studio</h1>
          <p>Planner · Executor · Evaluator</p>
        </div>
        <span className="status">● API Ready</span>
      </header>

      <section className="hero">
        <p className="eyebrow">PRACTICAL 09 · AGENTIC AI</p>
        <h2>Give your AI agent a goal.</h2>
        <p className="subtitle">
          Watch your multi-agent workflow plan, execute and evaluate tasks.
        </p>

        <div className="input-card">
          <label htmlFor="goal">What should your AI agent accomplish?</label>
          <textarea
            id="goal"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="Example: Explain artificial intelligence and give three real-world applications."
            rows={4}
          />
          <button onClick={runAgent} disabled={loading}>
            {loading ? "Running AI Agent..." : "▶ Run AI Agent"}
          </button>
          {error && <p className="error">{error}</p>}
        </div>
      </section>

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>AI agents are planning, executing and evaluating your goal...</p>
        </div>
      )}

      {result && (
        <section className="results">
          <h2>Workflow Evaluation</h2>

          <div className="metrics">
            <Metric title="Tasks Completed" value={`${result.tasks_completed}/${result.tasks_planned}`} />
            <Metric title="Completion Rate" value={`${(result.task_completion_rate * 100).toFixed(0)}%`} />
            <Metric title="Workflow Latency" value={`${result.workflow_latency_seconds}s`} />
            <Metric title="Quality Score" value={`${result.quality_score ?? "N/A"}/5`} />
            <Metric title="Accuracy Score" value={`${result.accuracy_score ?? "N/A"}/5`} />
          </div>

          <h3>Task Execution</h3>
          {result.task_results?.map((task, index) => (
            <article className="task-card" key={index}>
              <div className="task-heading">
                <strong>Task {index + 1}: {task.task}</strong>
                <span className={task.status === "Success" ? "success" : "failed"}>
                  {task.status}
                </span>
              </div>
              <p>{task.output || task.error || "No output available."}</p>
            </article>
          ))}

          <div className="feedback">
            <h3>LLM Judge Feedback</h3>
            <p>{result.judge_feedback || "No feedback available."}</p>
          </div>
        </section>
      )}
    </main>
  );
}

function Metric({ title, value }) {
  return (
    <div className="metric-card">
      <p>{title}</p>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
