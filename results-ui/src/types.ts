export type SweepRun = {
  scenario: string;
  depth: number;
  hidden: number;
  epochs: number;
  eml_final_val_loss: number;
  mlp_final_val_loss: number;
  eml_final_val_acc: number;
  mlp_final_val_acc: number;
  val_loss_gap_mlp_minus_eml: number;
  rel_dir: string;
  figures: Record<string, string>;
  eml_params?: number;
  mlp_params?: number;
};

export type SweepManifest = {
  depths: number[];
  hiddens: number[];
  preset?: string | null;
  scenarios: string[];
  epochs_default: number;
  runs: SweepRun[];
  heatmaps: { scenario: string; path: string }[];
};

export type LegacyRow = {
  scenario: string;
  epochs: number;
  depth: number;
  hidden: number;
  eml_final_val_loss: number;
  mlp_final_val_loss: number;
  eml_final_val_acc: number;
  mlp_final_val_acc: number;
  val_loss_gap_mlp_minus_eml: number;
  eml_params: number;
  mlp_params: number;
};
