import { FormEvent, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  askCase,
  createApplication,
  getCaseReportPdf,
  getDashboardStats,
  listApplications,
  submitDecision,
  uploadDocument,
  type DashboardStats,
  type DocumentUploadResponse,
  type LoanApplication,
  type LoanApplicationInput,
} from "./lib/api";

const initialForm: LoanApplicationInput = {
  applicant_name: "",
  applicant_email: "",
  employment_type: "salaried",
  monthly_income: 0,
  monthly_expenses: 0,
  existing_debts: 0,
  requested_amount: 0,
  tenure_months: 6,
  credit_history_years: 0,
  dependents: 0,
  purpose: "",
  notes: "",
  document_ids: [],
  transactions: [],
  document_checks: [],
};

function currency(value: number) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(
    value,
  );
}

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function App() {
  const [applications, setApplications] = useState<LoanApplication[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [selectedId, setSelectedId] = useState<string>("");
  const [form, setForm] = useState<LoanApplicationInput>(initialForm);
  const [question, setQuestion] = useState("Why was this case flagged?");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);
  const [decisionComment, setDecisionComment] = useState("");
  const [revisedAmount, setRevisedAmount] = useState("");
  const [revisedTenure, setRevisedTenure] = useState("");
  const [stagedUploads, setStagedUploads] = useState<DocumentUploadResponse[]>(
    [],
  );
  const [recentUploads, setRecentUploads] = useState<DocumentUploadResponse[]>(
    [],
  );
  const [statusMessage, setStatusMessage] = useState(
    "Upload PDFs or ID images to attach to the draft case or the currently selected case.",
  );

  const selectedApplication = useMemo(
    () =>
      applications.find((application) => application.id === selectedId) ??
      applications[0] ??
      null,
    [applications, selectedId],
  );

  async function refresh(preferredApplicationId?: string) {
    const [nextApplications, nextStats] = await Promise.all([
      listApplications(),
      getDashboardStats(),
    ]);
    setApplications(nextApplications);
    setStats(nextStats);
    const preferredApplication = nextApplications.find(
      (application) => application.id === preferredApplicationId,
    );
    if (preferredApplication) {
      setSelectedId(preferredApplication.id);
    } else if (!selectedId && nextApplications[0]) {
      setSelectedId(nextApplications[0].id);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    if (!selectedApplication) {
      return;
    }

    setRevisedAmount(String(selectedApplication.payload.requested_amount));
    setRevisedTenure(String(selectedApplication.payload.tenure_months));
  }, [selectedApplication]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!form.applicant_name.trim()) {
      setStatusMessage("Enter an applicant name before submitting the case.");
      return;
    }
    if (form.requested_amount <= 0 || form.tenure_months < 6) {
      setStatusMessage(
        "Enter a requested amount greater than zero and a tenure of at least 6 months.",
      );
      return;
    }
    setLoading(true);
    try {
      const created = await createApplication({
        ...form,
        document_ids: stagedUploads.map((upload) => upload.document.id),
      });
      setApplications((current) => [created, ...current]);
      setSelectedId(created.id);
      setStagedUploads([]);
      setStatusMessage(
        `Created case ${created.id} with ${created.documents.length} attached document(s).`,
      );
      await refresh(created.id);
    } catch (error) {
      setStatusMessage(
        error instanceof Error
          ? error.message
          : "Failed to create application.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleFileUpload(files: FileList | null) {
    if (!files || files.length === 0) {
      return;
    }

    setLoading(true);
    try {
      const uploads: DocumentUploadResponse[] = [];
      for (const file of Array.from(files)) {
        const response = await uploadDocument(file);
        uploads.push(response);
        setRecentUploads((current) => [response, ...current].slice(0, 6));
        setStagedUploads((current) => [response, ...current]);
      }
      setStatusMessage(
        `Staged ${uploads.length} file(s) for the next application.`,
      );
      await refresh();
    } catch (error) {
      setStatusMessage(
        error instanceof Error ? error.message : "Upload failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleDecision(
    decision: "approve" | "reject" | "modify" | "request_more_documents",
  ) {
    if (!selectedApplication) {
      return;
    }

    setLoading(true);
    try {
      const updated = await submitDecision(selectedApplication.id, {
        decision,
        comment: decisionComment,
        revised_amount: revisedAmount ? Number(revisedAmount) : undefined,
        revised_tenure_months: revisedTenure
          ? Number(revisedTenure)
          : undefined,
      });
      setSelectedId(updated.id);
      setStatusMessage(
        `Recorded ${decision} for ${updated.payload.applicant_name}.`,
      );
      if (decision === "request_more_documents") {
        const pdfBlob = await getCaseReportPdf(updated.id);
        const downloadUrl = window.URL.createObjectURL(pdfBlob);
        const anchor = document.createElement("a");
        anchor.href = downloadUrl;
        anchor.download = `Loan-IQ-Request-Docs-${updated.id}.pdf`;
        anchor.click();
        window.URL.revokeObjectURL(downloadUrl);
        setStatusMessage(
          `Recorded request docs for ${updated.payload.applicant_name} and downloaded the formal PDF.`,
        );
      }
      await refresh();
    } catch (error) {
      setStatusMessage(
        error instanceof Error ? error.message : "Decision update failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleAsk() {
    if (!selectedApplication) {
      return;
    }

    setLoading(true);
    try {
      const response = await askCase(selectedApplication.id, question);
      setAnswer(response.answer);
    } catch (error) {
      setAnswer(
        error instanceof Error ? error.message : "Unable to answer right now.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <section className="hero panel">
        <div>
          <p className="eyebrow">Loan IQ</p>
          <h1>
            The intelligent loan co-pilot for risk triage, uploads,
            explanations, and officer review.
          </h1>
          <p className="hero-copy">
            A multimodal hackathon scaffold that blends tabular scoring,
            cash-flow analysis, document authenticity checks, file upload
            handling, trainable model services, and human-in-the-loop decisions.
          </p>
        </div>
        <div className="hero-card">
          <span className="hero-kicker">Current queue</span>
          <strong>{stats?.review_queue ?? 0}</strong>
          <span>cases need officer attention</span>
        </div>
      </section>

      <section className="metrics">
        <MetricCard
          label="Total cases"
          value={stats?.total_cases ?? 0}
          detail="Submitted cases + new submissions"
        />
        <MetricCard
          label="Average score"
          value={percentage(stats?.average_final_score ?? 0)}
          detail="Higher means stronger repayment signal"
        />
        <MetricCard
          label="Overrides"
          value={percentage(stats?.override_rate ?? 0)}
          detail="Officer decisions already recorded"
        />
        <MetricCard
          label="Approved / Rejected"
          value={`${stats?.approved ?? 0} / ${stats?.rejected ?? 0}`}
          detail="Outcome balance in the queue"
        />
      </section>

      <section className="content-grid">
        <form className="panel form-panel" onSubmit={handleSubmit}>
          <div className="section-head">
            <h2>New application</h2>
            <p>
              Submit a loan request and attach uploaded PDFs or ID images before
              scoring.
            </p>
          </div>

          <div className="form-grid">
            <Field label="Applicant name">
              <input
                value={form.applicant_name}
                onChange={(event) =>
                  setForm({ ...form, applicant_name: event.target.value })
                }
              />
            </Field>
            <Field label="Email">
              <input
                value={form.applicant_email ?? ""}
                onChange={(event) =>
                  setForm({ ...form, applicant_email: event.target.value })
                }
              />
            </Field>
            <Field label="Employment type">
              <select
                value={form.employment_type}
                onChange={(event) =>
                  setForm({
                    ...form,
                    employment_type: event.target
                      .value as LoanApplicationInput["employment_type"],
                  })
                }
              >
                <option value="salaried">Salaried</option>
                <option value="self_employed">Self employed</option>
                <option value="business">Business</option>
                <option value="contract">Contract</option>
                <option value="other">Other</option>
              </select>
            </Field>
            <Field label="Monthly income">
              <input
                type="number"
                value={form.monthly_income}
                onChange={(event) =>
                  setForm({
                    ...form,
                    monthly_income: Number(event.target.value),
                  })
                }
              />
            </Field>
            <Field label="Monthly expenses">
              <input
                type="number"
                value={form.monthly_expenses}
                onChange={(event) =>
                  setForm({
                    ...form,
                    monthly_expenses: Number(event.target.value),
                  })
                }
              />
            </Field>
            <Field label="Existing debts">
              <input
                type="number"
                value={form.existing_debts}
                onChange={(event) =>
                  setForm({
                    ...form,
                    existing_debts: Number(event.target.value),
                  })
                }
              />
            </Field>
            <Field label="Requested amount">
              <input
                type="number"
                value={form.requested_amount}
                onChange={(event) =>
                  setForm({
                    ...form,
                    requested_amount: Number(event.target.value),
                  })
                }
              />
            </Field>
            <Field label="Tenure months">
              <input
                type="number"
                value={form.tenure_months}
                onChange={(event) =>
                  setForm({
                    ...form,
                    tenure_months: Number(event.target.value),
                  })
                }
              />
            </Field>
          </div>

          <Field label="Purpose">
            <input
              value={form.purpose}
              onChange={(event) =>
                setForm({ ...form, purpose: event.target.value })
              }
            />
          </Field>

          <Field label="Notes">
            <textarea
              rows={4}
              value={form.notes ?? ""}
              onChange={(event) =>
                setForm({ ...form, notes: event.target.value })
              }
            />
          </Field>

          <div className="upload-card">
            <div className="section-head compact">
              <h3>Document intake</h3>
              <p>
                Uploaded files are staged here and attached when you submit
                this new application.
              </p>
            </div>

            <div className="upload-grid">
              <Field label="Upload PDFs">
                <input
                  type="file"
                  accept="application/pdf,.pdf"
                  multiple
                  onChange={(event) =>
                    void handleFileUpload(event.target.files)
                  }
                />
              </Field>
              <Field label="Upload ID images">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={(event) =>
                    void handleFileUpload(event.target.files)
                  }
                />
              </Field>
            </div>

            <div className="tag-list">
              {stagedUploads.map((item) => (
                <span className="tag" key={item.document.id}>
                  {item.document.filename} · {item.document.document_kind}
                </span>
              ))}
              {!stagedUploads.length && (
                <span className="tag muted">No staged files</span>
              )}
            </div>
          </div>

          <button className="primary-button" type="submit" disabled={loading}>
            {loading ? "Working..." : "Run co-pilot triage"}
          </button>
        </form>

        <aside className="panel queue-panel">
          <div className="section-head">
            <h2>Officer queue</h2>
            <p>Sorted by latest submission in the backend dataset.</p>
          </div>

          <div className="upload-feed">
            <div className="section-head compact">
              <h3>Recent uploads</h3>
              <p>{statusMessage}</p>
            </div>
            {recentUploads.map((item) => (
              <div className="feed-row" key={item.document.id}>
                <strong>{item.document.filename}</strong>
                <span>{item.analysis}</span>
              </div>
            ))}
          </div>

          <div className="gemini-copilot">
            <div className="section-head compact">
              <h3>Ask Gemini</h3>
              <p>
                {selectedApplication
                  ? `Ask about ${selectedApplication.payload.applicant_name}'s case.`
                  : "Select a case from the queue to ask Gemini."}
              </p>
            </div>
            <textarea
              rows={4}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask about risk signals, cash flow, or documents"
              disabled={!selectedApplication}
            />
            <button
              className="primary-button"
              type="button"
              onClick={() => void handleAsk()}
              disabled={loading || !selectedApplication || !question.trim()}
            >
              Ask Gemini
            </button>
            <p className="answer-box">
              {answer ||
                "Gemini's explanation will appear here after you ask a question."}
            </p>
          </div>

          <div className="case-list">
            {applications.map((application) => (
              <button
                key={application.id}
                className={`case-row ${application.id === selectedApplication?.id ? "active" : ""}`}
                type="button"
                onClick={() => setSelectedId(application.id)}
              >
                <div>
                  <strong>{application.payload.applicant_name}</strong>
                  <span>{application.payload.purpose}</span>
                </div>
                <div className="case-meta">
                  <span
                    className={`badge ${application.scoring.risk_band.toLowerCase()}`}
                  >
                    {application.scoring.risk_band}
                  </span>
                  <span>{currency(application.payload.requested_amount)}</span>
                </div>
              </button>
            ))}
          </div>

        </aside>
      </section>

      {selectedApplication && (
        <section className="content-grid detail-grid">
          <article className="panel detail-panel">
            <div className="section-head">
              <h2>Case detail</h2>
              <p>
                {selectedApplication.payload.applicant_name} ·{" "}
                {selectedApplication.id}
              </p>
            </div>

            <div className="detail-topline">
              <span
                className={`badge ${selectedApplication.scoring.risk_band.toLowerCase()}`}
              >
                {selectedApplication.scoring.risk_band}
              </span>
              <span>{selectedApplication.scoring.recommendation}</span>
              <span>{selectedApplication.status}</span>
            </div>

            <div className="score-card">
              <div>
                <label>Final score</label>
                <strong>
                  {selectedApplication.scoring.final_score.toFixed(3)}
                </strong>
              </div>
              <div>
                <label>Confidence</label>
                <strong>
                  {percentage(selectedApplication.scoring.confidence)}
                </strong>
              </div>
            </div>

            <div className="score-bars">
              <ScoreBar
                label="Tabular"
                value={selectedApplication.scoring.tabular_score}
              />
              <ScoreBar
                label="Cash-flow"
                value={selectedApplication.scoring.cash_flow_score}
              />
              <ScoreBar
                label="Authenticity"
                value={selectedApplication.scoring.authenticity_score}
              />
            </div>

            <div className="split-grid">
              <section>
                <h3>Explainability</h3>
                <ul className="explain-list">
                  {selectedApplication.scoring.shap_explanation.map((item) => (
                    <li key={item.feature}>
                      <span>{item.feature}</span>
                      <strong
                        className={item.impact >= 0 ? "positive" : "negative"}
                      >
                        {item.impact > 0 ? "+" : ""}
                        {item.impact.toFixed(3)}
                      </strong>
                    </li>
                  ))}
                </ul>
              </section>

              <section>
                <h3>Monthly signal</h3>
                <ul className="explain-list">
                  {selectedApplication.scoring.time_step_signal.map((item) => (
                    <li key={item.month}>
                      <span>{item.month}</span>
                      <strong>{item.signal.toFixed(3)}</strong>
                    </li>
                  ))}
                </ul>
              </section>
            </div>

            <div className="decision-panel">
              <input
                className="inline-input"
                value={decisionComment}
                onChange={(event) => setDecisionComment(event.target.value)}
                placeholder="Officer comment"
              />
              <div className="decision-grid">
                <input
                  className="inline-input"
                  value={revisedAmount}
                  onChange={(event) => setRevisedAmount(event.target.value)}
                  placeholder="Revised amount"
                />
                <input
                  className="inline-input"
                  value={revisedTenure}
                  onChange={(event) => setRevisedTenure(event.target.value)}
                  placeholder="Revised tenure"
                />
              </div>
              <div className="button-row">
                <button
                  type="button"
                  onClick={() => void handleDecision("approve")}
                  disabled={loading}
                >
                  Approve
                </button>
                <button
                  type="button"
                  onClick={() => void handleDecision("reject")}
                  disabled={loading}
                >
                  Reject
                </button>
                <button
                  type="button"
                  onClick={() => void handleDecision("modify")}
                  disabled={loading}
                >
                  Modify
                </button>
                <button
                  type="button"
                  onClick={() => void handleDecision("request_more_documents")}
                  disabled={loading}
                >
                  Request docs
                </button>
              </div>
            </div>
          </article>
        </section>
      )}
    </main>
  );
}

function MetricCard({
  label,
  value,
  detail,
}: {
  label: string;
  value: number | string;
  detail: string;
}) {
  return (
    <div className="panel metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <p>{detail}</p>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function ScoreBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="score-bar">
      <div className="score-bar-labels">
        <span>{label}</span>
        <strong>{value.toFixed(3)}</strong>
      </div>
      <div className="progress-track">
        <div
          className="progress-fill"
          style={{ width: `${Math.round(value * 100)}%` }}
        />
      </div>
    </div>
  );
}

export default App;
