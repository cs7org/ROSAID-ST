import argparse
from pathlib import Path

import torch
from art.attacks.evasion import (MomentumIterativeMethod, ProjectedGradientDescent,Wasserstein, PixelAttack)
from art.estimators.classification import PyTorchClassifier
from loguru import logger

from rosaid.adversarial import AdversarialExperiment, ClfModel
from rosaid.data.dataset import (DFDataSet, SessionImageDataConfig,
                                   TorchImageDataset)

# parser for model_names by comma separated, batch_size
# data_dir, project_dir, image_type
parser = argparse.ArgumentParser(
    description="Adversarial Image Generation Configuration"
)
parser.add_argument(
    "--model_names",
    type=str,
    default="resnet18_nosampling,mobilenet_v3_large_nosampling",
    help="Comma-separated list of model names to use for adversarial attacks.",
)
parser.add_argument(
    "--image_type",
    type=str,
    choices=["normal", "normalized"],
    help="Type of images to use for training. 'normal' for raw images, 'normalized' for normalized images.",
)
parser.add_argument(
    "--max_data",
    type=int,
    default=100,
    help="Maximum number of data points to use. Use -ve for all data.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    help="Batch size for adversarial attack generation.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=r"data\120_timeout_dnp3_sessions",
    help="Directory containing the dataset.",
)
parser.add_argument(
    "--project_dir",
    type=str,
    default=r".",
    help="Directory containing the project files.",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default=r"data",
    help="Directory saving the generated files.",
)
parser.add_argument(
    "--epsilons",
    type=str,
    # default="0.001,0.01,0.1,0.2,0.3,0.5",
    default="0.7,0.9",
    help="Comma-separated list of epsilon values for adversarial attacks.",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=10,
    help="Number of iterations for iterative adversarial attacks.",
)
# flag to run adv or not
parser.add_argument(
    "--run_adv",
    action="store_true",
    help="Flag to indicate whether to run adversarial attacks or not.",
)
parser.add_argument(
    "--validation_only",
    action="store_true",
    default=True,
    help="Flag to indicate whether to run validation only or not.",
)


args = parser.parse_args()

save_dir = args.save_dir
targeted = True
save_dir = save_dir if not targeted else save_dir + "_targeted"

# Parse arguments
model_names = [m.strip() for m in args.model_names.split(",")]
project_dir = Path(args.project_dir)
save_dir = Path(save_dir)
data_dir = Path(args.data_dir)
if not data_dir.exists():
    logger.error(f"Data directory does not exist: {data_dir}")
    raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
if not project_dir.exists():
    logger.error(f"Project directory does not exist: {project_dir}")
    raise FileNotFoundError(f"Project directory does not exist: {project_dir}")
epsilons = [float(eps.strip()) for eps in args.epsilons.split(",") if eps.strip()]
# sort epsilons in descending order
epsilons.sort(reverse=True)
batch_size = args.batch_size
validation_only = args.validation_only

# !!!IMPORTANT: full model might not be usable when package is not installed
model_paths = [
    project_dir / "results" / "image_classification" / name / "best_model_full.pth"
    for name in model_names
]
iterations = args.iterations
max_data = args.max_data
use_normalized = args.image_type.lower() == "normalized"

for model_path in model_paths:
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    logger.info(f"Model path exists: {model_path}")
    config = SessionImageDataConfig(
        max_data=max_data,
        session_images_dir=data_dir / "session_images",
        labels_file=data_dir / "labelled_sessions.csv",
        use_normalized=use_normalized,
        num_pkts=100,
        byte_length=256,
        combine_attacks=False,
        balanced_batches=False
    )
    train_ds, test_ds = DFDataSet(config=config).load_data()
    model_path = model_path.resolve()
    # this might fail if package is not installed
    logger.info(f"Loading model from: {model_path}")
    model = torch.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=False,
    )

    logger.info(f"Running adversarial attacks on model: {model_path.parent.name}")

    iterations = 50
    input_shape = (1, config.num_pkts, config.byte_length)
    clf_model = ClfModel(model)
    
    loss = torch.nn.BCELoss() if model.output_size == 1 else torch.nn.CrossEntropyLoss()

    clf_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    nb_classes = len(train_ds.label_encoding) if 'binary' not in model_path.parent.name.lower() else 2
    attacks=[]
    
    attacks.extend([
        ProjectedGradientDescent(
            estimator=PyTorchClassifier(
                model=clf_model,
                loss=loss,
                clip_values=(0, 1),
                input_shape=input_shape,
                nb_classes=nb_classes,
                optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            ),
            eps=eps,
            batch_size=batch_size,
            targeted=targeted,
            verbose=False,  
        )
        for eps in epsilons
    ])
    
    
    attacks.extend(
        [
            MomentumIterativeMethod(
                estimator=PyTorchClassifier(
                    model=ClfModel(model),
                    loss=loss,
                    clip_values=(0, 1),
                    input_shape=input_shape,
                    nb_classes=nb_classes,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                ),
                eps=eps,
                targeted=targeted,
                batch_size=batch_size,
                verbose=False,
                max_iter=iterations,

            )
            for eps in epsilons
        ]
    )
    attacks.extend(
        [
            PixelAttack(
                classifier=PyTorchClassifier(
                    model=ClfModel(model),
                    loss=loss,
                    clip_values=(0, 1),
                    input_shape=input_shape,
                    nb_classes=nb_classes,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                ),
                targeted=False,
                verbose=False,
                max_iter=iterations,

            )
            
        ]
    )

    attacks.extend(
        [
            Wasserstein(
                estimator=PyTorchClassifier(
                    model=ClfModel(model),
                    loss=loss,
                    clip_values=(0, 1),
                    input_shape=input_shape,
                    nb_classes=nb_classes,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
                ),
                eps=eps,
                targeted=targeted,
                batch_size=batch_size,
                verbose=False,
                max_iter=iterations,

            )
            for eps in epsilons
        ]
    )

    

    # attacks.extend(
    #     [
    #         # CarliniL2Method(
    #         #     classifier=PyTorchClassifier(
    #         #         model=ClfModel(model),
    #         #         loss=loss,
    #         #         clip_values=(0, 1),
    #         #         input_shape=input_shape,
    #         #         nb_classes=nb_classes,
    #         #         optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    #         #     ),
    #         #     targeted=targeted,
    #         #     batch_size=batch_size,
    #         #     verbose=False,
    #         #     max_iter=2,
    #         # ),
    #         # DeepFool(
    #         #     classifier=PyTorchClassifier(
    #         #         model=ClfModel(model),
    #         #         loss=loss,
    #         #         clip_values=(0, 1),
    #         #         input_shape=input_shape,
    #         #         nb_classes=nb_classes,
    #         #         optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    #         #     ),
    #         #     batch_size=batch_size,
    #         #     verbose=False,
    #         # ),
    #         # SaliencyMapMethod(
    #         #     classifier=PyTorchClassifier(
    #         #         model=ClfModel(model),
    #         #         loss=loss,
    #         #         clip_values=(0, 1),
    #         #         input_shape=input_shape,
    #         #         nb_classes=nb_classes,
    #         #         optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    #         #     ),
    #         #     batch_size=batch_size,
    #         #     verbose=False,
    #         # ),
    #     ]
    # )

    adv = AdversarialExperiment(
        model=model,
        model_name=model_path.parent.name,
        attacks=attacks,
        train_dataset=TorchImageDataset(train_ds),
        test_dataset=TorchImageDataset(test_ds),
        input_shape=input_shape,
        output_dir=data_dir / "adversarial_attacks" / model_path.parent.name,
        batch_size=batch_size,
        targeted=targeted,
       
    )

    if not args.run_adv:
        logger.info("Skipping adversarial attack evaluation as --run_adv is not set.")
    else:
        adv.run(
            results_dir=(
                project_dir / "results" / "adversarial_attacks" / model_path.parent.name
            )
        )
        logger.info("Adversarial attacks completed successfully.")
    logger.info("Generating adversarial attack data...")

    selected_attacks = []
    for atk in attacks:
        if not hasattr(atk, "eps"):
            atk.eps = 0.0
            selected_attacks.append(atk)
        elif atk.eps in epsilons:
            selected_attacks.append(atk)

    for attack in selected_attacks:
        logger.info(
            f"Generating adversarial data for attack: {attack.__class__.__name__} with eps: {attack.eps}"
        )
        out_folder = attack.__class__.__name__.lower() + f"_eps_{attack.eps}"
        copy_compressed_to = (
            save_dir
            / "results"
            / "adversarial_attacks"
            / model_path.parent.name
            / out_folder
        )
        try:
            if validation_only:
                logger.info("Running validation only for adversarial data generation.")
            adv.generate(
                attack,
                out_folder=out_folder,
                copy_compressed_to=copy_compressed_to,
                validation_only=validation_only,
            )
        except Exception as e:
            import traceback
            logger.error(
                f"Error generating adversarial data for attack {attack.__class__.__name__} with eps {attack.eps}: {e}"
            )
            logger.error(traceback.format_exc())

    logger.info("Adversarial image generation completed successfully.")
