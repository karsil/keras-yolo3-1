from keras.backend import clear_session
import os
import json
import optuna
import keras
import argparse
from utils.utils import normalize, evaluate, makedirs
from train import create_callbacks, create_training_instances, create_model
from generator import BatchGenerator
from tqdm.keras import TqdmCallback
from datetime import datetime
from pathlib import Path

study_name = "optuna_results_21sep"
config = None

def create_optimizer(trial):
    kwargs = {}
    optimizer_options = ['Adam', 'RMSprop', 'SGD']
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-2)
        kwargs["beta_1"] = trial.suggest_float("beta1", 0.9, 0.9999)
        kwargs["beta_2"] = trial.suggest_float("beta2", 0.9, 0.9999)
        kwargs["epsilon"] = trial.suggest_float("epsilon", 1e-9, 1)
    elif optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rms_learning_rate", 1e-7, 1e-2
        )
        kwargs["decay"] = trial.suggest_float("rms_decay", 0.8, 0.9999)
        kwargs["epsilon"] = trial.suggest_float("rms_epsilon", 1e-12, 1e-7)
        kwargs["momentum"] = trial.suggest_float("rms_momentum", 0, 1e-1)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-9, 1e-1
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1)
        kwargs["nesterov"] = trial.suggest_categorical('sgd_nosterov', [True, False])
    else:
        assert False, "ERROR: Got {} as optimizer".format(optimizer_selected)
    
    optimizer = getattr(keras.optimizers, optimizer_selected)(**kwargs)
    return optimizer, kwargs["learning_rate"]


def objective(trial):
    clear_session()
    batch_size = trial.suggest_categorical("batch_size", [2,3,4])
    optimizer, lr = create_optimizer(trial)

    result_path = "studies"
    Path(result_path).mkdir(parents=True, exist_ok=True)
    log_dir = result_path + '/' + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + '/'

    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################
    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels'],
        config['train']['train_val_split']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

    ###############################
    #   Create the generators 
    ###############################    
    train_generator = BatchGenerator(
        instances           = train_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = batch_size,
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.3, 
        norm                = normalize
    )
    
    valid_generator = BatchGenerator(
        instances           = valid_ints, 
        anchors             = config['model']['anchors'],   
        labels              = labels,        
        downsample          = 32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image   = max_box_per_image,
        batch_size          = config['train']['batch_size'],
        min_net_size        = config['model']['min_input_size'],
        max_net_size        = config['model']['max_input_size'],   
        shuffle             = True, 
        jitter              = 0.0, 
        norm                = normalize
    )

    if os.path.exists(config['train']['saved_weights_name']): 
        config['train']['warmup_epochs'] = 0
    warmup_batches = config['train']['warmup_epochs'] * (config['train']['train_times']*len(train_generator))   

    #os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    #multi_gpu = len(config['train']['gpus'].split(','))
    #os.environ['CUDA_VISIBLE_DEVICES'] = 4
    multi_gpu = 1

    train_model, infer_model = create_model(
        nb_class            = len(labels), 
        anchors             = config['model']['anchors'], 
        max_box_per_image   = max_box_per_image, 
        max_grid            = [config['model']['max_input_size'], config['model']['max_input_size']], 
        batch_size          = batch_size, 
        warmup_batches      = warmup_batches,
        ignore_thresh       = config['train']['ignore_thresh'],
        multi_gpu           = 1,
        saved_weights_name  = config['train']['saved_weights_name'],
        lr                  = lr,
        grid_scales         = config['train']['grid_scales'],
        obj_scale           = config['train']['obj_scale'],
        noobj_scale         = config['train']['noobj_scale'],
        xywh_scale          = config['train']['xywh_scale'],
        class_scale         = config['train']['class_scale'],
        optimizer=optimizer
    )

    epochs = config['train']['nb_epochs'] + config['train']['warmup_epochs']

    target_weight = log_dir + config['train']['saved_weights_name']
    tb_dir = log_dir + config['train']['tensorboard_dir']
    callbacks = create_callbacks(target_weight, tb_dir, infer_model)
    callbacks = callbacks + [
        optuna.integration.TFKerasPruningCallback(
            trial=trial,
            monitor='loss'
        ),
        TqdmCallback(epochs=epochs, batch_size=batch_size)
    ]
    
    history = train_model.fit_generator(
        generator        = train_generator, 
        steps_per_epoch  = len(train_generator) * config['train']['train_times'], 
        epochs           = epochs, 
        verbose          = 2 if config['train']['debug'] else 1,
        callbacks        = callbacks, 
        workers          = 1,
        max_queue_size   = 8
    )

    print(f"Study: {log_dir}")
    for key in history.history.keys():
        print(key, '->', history.history[key])

    last_loss = float(history.history['loss'][-1])
    print(last_loss)
    return last_loss

def _main_(args):
    study = optuna.create_study(
        direction="minimize",
        study_name = study_name,
        sampler = optuna.samplers.TPESampler(),
        pruner = optuna.pruners.MedianPruner(),
        storage = "sqlite:///" + study_name + ".db",
        load_if_exists = True
    )
    study.optimize(objective, n_trials=10, timeout=600)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    results = study.trials_dataframe()
    results.to_csv(study_name + ".csv")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', default='config.json', help='path to configuration file')   

    args = argparser.parse_args()
    _main_(args)
