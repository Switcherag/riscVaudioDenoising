import nncase

print("eeeeeeeeeeeeeeeeeeee")
def read_model_file(model_file):
    with open(model_file, 'rb') as f:
        model_content = f.read()
    return model_content


model='C:/Users/Adam/Desktop/riscVaudioDenoising/DeepLearning/your_model.tflite'
target = 'k510'

# compile_options
compile_options = nncase.CompileOptions()
compile_options.target = target
compile_options.dump_ir = True
compile_options.dump_asm = True
compile_options.dump_dir = 'tmp'  
compile_options.input_type = 'float32'  # or 'uint8' 'int8'
compile_options.input_shape = [None, 128, 128, 1]  # keep layout same as input layout



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
with open('test.kmodel', 'wb') as f:
    f.write(kmodel)

