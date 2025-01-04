----api/
    cv/
        img_processing.py          # Image processing functions for CV tasks
        research.ipynb             # Jupyter notebook for research analysis
        runTiles.py                # Script for running tile-based operations
        yolov8.py                  # YOLOv8 implementation for object detection
    data/
        output/                    # Folder for generated output files
        tiles/                     # Folder for image tiles
    utils/
        helpers.py                 # Helper functions used across modules
    main.py                        # Main script to run the application
----datasets/                      # Folder containing dataset files
----models/
    flower_model.pt                # Model for flower classification
    reference_object_model.pt      # Model for reference object detection
----research/
    data/
        evaluation_images_experiment_1/ # Images for evaluation in experiment 1
        experiment_3_images/            # Images for experiment 3
        floralarea_experiment_1.csv     # Results of floral area experiment 1
        measuring_time_experiment3.csv  # Timing data for experiment 3
    data_analysis.ipynb            # Notebook for data analysis
----scripts/
    data_processing/
        download_roboflow_data.py  # Script to download data from Roboflow
    estimator/
        FloralArea.py              # Estimation logic for floral area
    training/
        evaluate.py                # Evaluation script for models
        test.py                    # Testing script for models
        train.py                   # Training script for models
    update_settings.py             # Script for updating project settings
.gitignore                         # Files and directories to ignore in Git
README.md                          # This README file
requirements.txt                   # Python dependencies
