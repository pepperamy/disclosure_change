import config
import trainer
import utils
import simple_siamese_model
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":

    myconfig = config.config()
    myconfig.tokenize()
    myconfig.bert()
    myconfig.load_data()
    
    mymodel = simple_siamese_model.simple_siamese(myconfig)

    if myconfig.test_mode == True:
        myconfig.test_model(mymodel, myconfig)
    else:
        res = trainer.training_loop(myconfig, mymodel)