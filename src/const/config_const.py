
class GENERAL_TXT:
    GENERAL = "GENERAL"

    input_dir = "input_dir"

    train_dir = "train_dir"
    train_split = "train_split"
    train_annos = "train_annos"
    val_dir = "val_dir"
    val_split = "val_split"
    val_annos = "val_annos"
    save_labels_dir = "save_labels_dir"

    image_extensions = "image_extensions"
    batch_size = "batch_size"
    plot_count = "plot_count"
    input_annos = "input_annos"
    quiet = "quiet"


class ANALYZE_FREQ:
    ANALYZE_FREQ = "ANALYZE_FREQ"

    r_values = "r_values"


class EXPERIMENTS:
    EXPERIMENTS = "EXPERIMENTS"

    exp_number = "exp_number"
    exp_values = "exp_values"
    force_exp = "force_exp"
    plot_analyze = "plot_analyze"
    force_detect = "force_detect"


class MODEL_TXT:
    MODEL = "MODEL"

    model_type = "model_type"
    model_weights = "model_weights"
    score_threshold = "score_threshold"


MODEL_TYPES = ["3",
               "5n", "5s", "5m", "5l", "5x",
               "8n", "8s", "8m", "8l", "8x"]
