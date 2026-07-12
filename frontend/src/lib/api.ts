export interface TransactionSummary {
  month: string;
  inflow: number;
  outflow: number;
  closing_balance: number;
}

export interface DocumentCheck {
  document_type: string;
  authenticity_confidence: number;
  face_match_similarity: number;
  tamper_flag: boolean;
}

export interface LoanApplicationInput {
  applicant_name: string;
  applicant_email?: string;
  employment_type:
    | "salaried"
    | "self_employed"
    | "business"
    | "contract"
    | "other";
  monthly_income: number;
  monthly_expenses: number;
  existing_debts: number;
  requested_amount: number;
  tenure_months: number;
  credit_history_years: number;
  dependents: number;
  purpose: string;
  notes?: string;
  document_ids: string[];
  transactions: TransactionSummary[];
  document_checks: DocumentCheck[];
}

export interface UploadedDocument {
  id: string;
  application_id: string | null;
  filename: string;
  content_type: string;
  document_kind: "pdf" | "id_image";
  size_bytes: number;
  sha256: string;
  byte_entropy: number;
  ascii_ratio: number;
  created_at: string;
}

export interface DocumentUploadResponse {
  document: UploadedDocument;
  model_score: number;
  confidence: number;
  analysis: string;
}

export interface FeatureAttribution {
  feature: string;
  contribution: number;
}

export interface ModelPrediction {
  model_name: string;
  model_version: string;
  score: number;
  confidence: number;
  attributions: FeatureAttribution[];
}

export interface ScoringOutput {
  final_score: number;
  risk_band: string;
  recommendation: string;
  confidence: number;
  fusion_weights: Record<string, number>;
  tabular_score: number;
  cash_flow_score: number;
  authenticity_score: number;
  model_outputs: ModelPrediction[];
  shap_explanation: Array<{ feature: string; impact: number }>;
  time_step_signal: Array<{ month: string; signal: number }>;
  document_rationale: string;
}

export interface LoanApplication {
  id: string;
  submitted_at: string;
  payload: LoanApplicationInput;
  scoring: ScoringOutput;
  documents: UploadedDocument[];
  officer_decision: string | null;
  officer_comment: string | null;
  status: string;
}

export interface DashboardStats {
  total_cases: number;
  review_queue: number;
  approved: number;
  rejected: number;
  average_final_score: number;
  override_rate: number;
}

export interface DecisionRequest {
  decision: "approve" | "reject" | "modify" | "request_more_documents";
  comment?: string;
  revised_amount?: number;
  revised_tenure_months?: number;
}

export interface ChatResponse {
  answer: string;
}

export interface ModelStatus {
  model_name: string;
  model_version: string;
  trained: boolean;
  sample_count: number;
  accuracy: number | null;
  trained_at: string | null;
}

export interface ModelTrainRequest {
  include_seed_cases: boolean;
  include_unattached_documents: boolean;
}

export interface ModelTrainResponse {
  trained_at: string;
  model_statuses: ModelStatus[];
  sample_count: number;
}

const API_BASE = import.meta.env.VITE_API_URL ?? "";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }

  return (await response.json()) as T;
}

async function requestText(path: string): Promise<string> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }
  return await response.text();
}

async function requestBlob(path: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }
  return await response.blob();
}

export async function listApplications(): Promise<LoanApplication[]> {
  return requestJson("/api/applications");
}

export async function createApplication(
  input: LoanApplicationInput,
): Promise<LoanApplication> {
  return requestJson("/api/applications", {
    method: "POST",
    body: JSON.stringify(input),
  });
}

export async function uploadDocument(
  file: File,
  applicationId?: string,
): Promise<DocumentUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (applicationId) {
    formData.append("application_id", applicationId);
  }

  const response = await fetch(`${API_BASE}/api/uploads/documents`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with ${response.status}`);
  }

  return (await response.json()) as DocumentUploadResponse;
}

export async function listDocuments(): Promise<DocumentUploadResponse[]> {
  return requestJson("/api/uploads/documents");
}

export async function getModelStatus(): Promise<ModelStatus[]> {
  return requestJson("/api/models/status");
}

export async function trainModels(
  request: ModelTrainRequest = {
    include_seed_cases: true,
    include_unattached_documents: false,
  },
): Promise<ModelTrainResponse> {
  return requestJson("/api/models/train", {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export async function submitDecision(
  applicationId: string,
  decision: DecisionRequest,
): Promise<LoanApplication> {
  return requestJson(`/api/applications/${applicationId}/decision`, {
    method: "POST",
    body: JSON.stringify(decision),
  });
}

export async function getCaseReport(applicationId: string): Promise<string> {
  return requestText(`/api/applications/${applicationId}/report`);
}

export async function getCaseReportPdf(applicationId: string): Promise<Blob> {
  return requestBlob(`/api/applications/${applicationId}/report/pdf`);
}

export async function askCase(
  applicationId: string,
  question: string,
): Promise<ChatResponse> {
  return requestJson(`/api/applications/${applicationId}/chat`, {
    method: "POST",
    body: JSON.stringify({ question }),
  });
}

export async function getDashboardStats(): Promise<DashboardStats> {
  return requestJson("/api/dashboard/stats");
}
