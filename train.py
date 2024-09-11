from src.train.train_models import train_model
from src.utils.plot_metrics import plot_metrics
import itertools

def start_train():
    # Defined lists of hyperparameters
    epochs_list = [20]
    lr_list = [0.001]
    batch_size_list = [2]
    loss_functions = ['BCEWithLogitsLoss']

    # Defined list of models
    models = [
        ('deeplabv3_resnet50', loss_functions),
        ('fcn_resnet50', loss_functions),
        ('custom_unet', loss_functions)
    ]

    # Plot data
    plot_data = {epoch: {'train_losses': [], 'val_losses': [], 'val_iou_scores': [], 'val_f1_scores': []} for epoch in epochs_list}

    # Open file for writing results
    with open('results/training_results_run3.txt', 'w') as file:
        # Iterating through all models
        for model_name, loss_options in models:
            # Iterating through all hyperparameters
            for epoch, lr, batch_size, loss_function in itertools.product(epochs_list, lr_list, batch_size_list,
                                                                          loss_options):
                # Training the model
                train_losses, val_losses, val_iou_scores, val_f1_scores = train_model(
                    epochs=epoch,
                    batch_size=batch_size,
                    lr=lr,
                    model_name=model_name,
                    criterion_name=loss_function
                )

                # Defining configuration key
                config_key = f'{model_name} Epochs={epoch} LR={lr} BS={batch_size} Loss={loss_function}'

                # Writing results to file
                file.write(f'Configuration: {config_key}\n')
                file.write('Training Losses: ' + str(train_losses) + '\n')
                file.write('Validation Losses: ' + str(val_losses) + '\n')
                file.write('Validation IoU Scores: ' + str(val_iou_scores) + '\n')
                file.write('Validation F1 Scores: ' + str(val_f1_scores) + '\n')
                file.write('-----------------------------------\n')

                # Appending results to plot data
                plot_data[epoch]['train_losses'].append({'name': config_key, 'data': train_losses})
                plot_data[epoch]['val_losses'].append({'name': config_key, 'data': val_losses})
                plot_data[epoch]['val_iou_scores'].append({'name': config_key, 'data': val_iou_scores})
                plot_data[epoch]['val_f1_scores'].append({'name': config_key, 'data': val_f1_scores})

    for epochs in epochs_list:
        # Plotting metrics
        plot_metrics(
            metrics=plot_data[epochs]['train_losses'],
            title='Training Loss Comparison',
            x_label='Epochs',
            y_label='Loss'
        )

        plot_metrics(
            metrics = plot_data[epochs]['val_losses'],
            title = 'Validation Loss Comparison',
            x_label = 'Epochs',
            y_label = 'Loss'
        )

        plot_metrics(
            metrics = plot_data[epochs]['val_iou_scores'],
            title = 'Validation IoU Scores Comparison',
            x_label = 'Epochs',
            y_label = 'IoU Score'
        )

        plot_metrics(
            metrics = plot_data[epochs]['val_f1_scores'],
            title = 'Validation F1 Scores Comparison',
            x_label = 'Epochs',
            y_label = 'F1 Score'
        )
