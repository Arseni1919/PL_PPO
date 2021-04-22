from CONSTANTS import *
from alg_lightning_module import ALGLightningModule
from alg_datamodule import ALGDataModule
from alg_callbaks import ALGCallback
from try_weights import play
from help_functions import *


def main():
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    model = ALGLightningModule()
    data_module = ALGDataModule(net=model.net)

    trainer = pl.Trainer(callbacks=[ALGCallback()],
                         max_epochs=MAX_EPOCHS,
                         val_check_interval=VAL_CHECKPOINT_INTERVAL,
                         logger=tb_logger)

    trainer.fit(model=model,
                datamodule=data_module)

    plot_('training loss', model.log_for_loss)
    play(NUMBER_OF_GAMES)


if __name__ == '__main__':
    main()

    # to run tensorboard:
    # tensorboard --logdir lightning_logs

