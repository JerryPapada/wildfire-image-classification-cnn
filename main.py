from model_functions import build_model, compile_model, train_model, create_generators, evaluate_model, modify_top_layers_for_binary_classification
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, MobileNetV2, DenseNet121
from telegram import Bot
import asyncio
import hyperparameters as hp

async def send_telegram_message(bot_token, chat_id, message):
    bot = Bot(token=bot_token)
    await bot.send_message(chat_id=chat_id, text=message)

async def main():
    #SATELLITE DATASET
    train = "/home/gerpapa/Wildfire-Classification-CNN/dataset/train"
    validation = "/home/gerpapa/Wildfire-Classification-CNN/dataset/valid"
    test = "/home/gerpapa/Wildfire-Classification-CNN/dataset/test"

    # generators are universal for both pretrained and custom CNNs
    train_generator, validation_generator, test_generator = create_generators(train, validation, test)

    #===== CUSTOM CNN ====#
    model = build_model(input_shape=(hp.img_height, hp.img_width, 3))
    model = compile_model(model)
    
    results = train_model(model, train_generator, validation_generator, epochs=hp.epochs)
    ml_results = evaluate_model(model,test_generator)
    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results)
    #=====================#
    '''
    # Load pre-trained models
    vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(hp.img_height, hp.img_width, 3))
    vgg19_model = VGG19(weights='imagenet', include_top=False, input_shape=(hp.img_height, hp.img_width, 3))
    resnet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(hp.img_height, hp.img_width, 3))
    inception_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(hp.img_height, hp.img_width, 3))
    mobilenetv2_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(hp.img_height, hp.img_width, 3))
    densenet121_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(hp.img_height, hp.img_width, 3))

    # Modify top layers for binary classification
    vgg16_model = modify_top_layers_for_binary_classification(vgg16_model)
    vgg19_model = modify_top_layers_for_binary_classification(vgg19_model)
    resnet50_model = modify_top_layers_for_binary_classification(resnet50_model)
    inception_model = modify_top_layers_for_binary_classification(inception_model)
    mobilenetv2_model = modify_top_layers_for_binary_classification(mobilenetv2_model)
    densenet121_model = modify_top_layers_for_binary_classification(densenet121_model)

    # Compile models
    compile_model(vgg16_model)
    compile_model(vgg19_model)
    compile_model(resnet50_model)
    compile_model(inception_model)
    compile_model(mobilenetv2_model)
    compile_model(densenet121_model)

    # Train models
    results_vgg16 = train_model(vgg16_model, train_generator, validation_generator, epochs=hp.epochs)
    results_vgg19 = train_model(vgg19_model, train_generator, validation_generator, epochs=hp.epochs)
    results_resnet50 = train_model(resnet50_model, train_generator, validation_generator, epochs=hp.epochs)
    results_inception = train_model(inception_model, train_generator, validation_generator, epochs=hp.epochs)
    results_mobilenetv2 = train_model(mobilenetv2_model, train_generator, validation_generator, epochs=hp.epochs)
    results_densenet121 = train_model(densenet121_model, train_generator, validation_generator, epochs=hp.epochs)

    # Evaluate models
    ml_results_vgg16 = evaluate_model(vgg16_model, test_generator)
    ml_results_vgg19 = evaluate_model(vgg19_model, test_generator)
    ml_results_resnet50 = evaluate_model(resnet50_model, test_generator)
    ml_results_inception = evaluate_model(inception_model, test_generator)
    ml_results_mobilenetv2 = evaluate_model(mobilenetv2_model, test_generator)
    ml_results_densenet121 = evaluate_model(densenet121_model, test_generator)

    # Telegram notification
    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results_vgg16)
    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results_vgg19)
    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results_resnet50)
    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results_inception)
    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results_mobilenetv2)
    await send_telegram_message("6324537709:AAHjIuf4fEtFexsv8q3kh6o7oQdmg9zJ_qc", "6416416336", ml_results_densenet121)
    '''
if __name__ == "__main__":
    asyncio.run(main())
