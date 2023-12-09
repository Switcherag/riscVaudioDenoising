import nncase
import tensorflow as tf
from tensorflow.keras.models import model_from_json

print("eeeeeeeeeeeeeeeeeeee")
def read_model_file(model_file):
    
    # load json and create model
    json_file = open('./weights/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model/
    loaded_model.load_weights('./weights/model.h5')
    print("Loaded model from disk")

    # --------------------------------------Create a TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

    #Perform the conversion to TFLite format
    tflite_model = converter.convert()

    return tflite_model


model='./weights/your_model.tflite'


# compile_options
compile_options = nncase.CompileOptions()
compile_options.target = "k210"
compile_options.dump_ir = True
compile_options.dump_asm = True
compile_options.dump_dir = './MachineLearning/tmp'  
compile_options.input_type = 'float32'  # or 'uint8' 'int8'



# compiler
compiler = nncase.Compiler(compile_options)

# import_options
import_options = nncase.ImportOptions()

# import
model_content = read_model_file(model)
compiler.import_tflite(model_content, import_options)

# compile
compiler.compile()

# kmodel
kmodel = compiler.gencode_tobytes()
with open('./weights/test.kmodel', 'wb') as f:
    f.write(kmodel)

