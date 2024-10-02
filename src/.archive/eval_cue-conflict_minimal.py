import os
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50
from pytorch_lightning import seed_everything

from probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping
from models import ModelFactory


DATA_PATH = '/evaluation/cue-conflict/texture-vs-shape/stimuli/style-transfer-preprocessed-512'
CATEGORIES = ["airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
              "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck"]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
mapping = ImageNetProbabilitiesTo16ClassesMapping()


def evaluate_model_texture_bias(model, verbose=False):
    """
    following the paper, we only count the Texture Bias by counting correct predictions
    and then dividing
    """

    shape_preds = {category: 0 for category in CATEGORIES}
    texture_preds = {category: 0 for category in CATEGORIES}
    shape_preds_total = 0
    texture_preds_total = 0

    # iterate over all categories and images
    for category in CATEGORIES:
        category_path = f"{DATA_PATH}/{category}"
        category_image_filenames = os.listdir(category_path)
        for category_image_filename in category_image_filenames:
            # get texture name from image name
            texture = category_image_filename.split('-')[1].replace('.png', '')
            texture = ''.join([i for i in texture if not i.isdigit()])
            # skip if no cue-conflict
            if texture == category:
                continue
            # load image and predict category out of 16 classes
            cue_conflict_image = Image.open(f"{category_path}/{category_image_filename}")
            normalized_input = transform(cue_conflict_image)
            normalized_input = normalized_input.unsqueeze(0)  # add batch dimension
            output = model(normalized_input)
            output = output.squeeze(0)  # remove batch dimension
            softmax_output = output.softmax(dim=0).detach().numpy()
            predicted_category = mapping.probabilities_to_decision(softmax_output)

            # check if predicted is either category or texture, else skip (according to the paper)
            if not (predicted_category == category or predicted_category == texture):
                continue
            # count correct predictions
            if predicted_category == category:
                shape_preds[category] += 1
                shape_preds_total += 1
            elif predicted_category == texture:
                texture_preds[category] += 1
                texture_preds_total += 1
            if verbose:
                print(f"[ {category:<8} | {texture:<8} ]:  {predicted_category}")

    # calculate texture bias, both total and per category
    texture_bias_total = texture_preds_total / (texture_preds_total + shape_preds_total)
    texture_bias = {category: texture_preds[category] / (texture_preds[category] + shape_preds[category])
                    for category in CATEGORIES}
    texture_bias_avg = sum(texture_bias.values()) / len(texture_bias)

    # log results
    print(f"Texture-Bias Total: {texture_bias_total * 100:.0f} %")
    print(f"Texture-Bias Average: {texture_bias_avg * 100:.0f} %")
    print("Texture-Bias per Category:")
    for category in CATEGORIES:
        print(f"\t{category:<8}: {texture_bias[category] * 100:.0f} %")

    results = {
        'total': texture_bias_total,
        'average': texture_bias_avg,
    }


    return results


if __name__ == '__main__':
    seed_everything(42)
    model = resnet50(weights='IMAGENET1K_V1')  # use the default Imagenet_1k weights
    model2 = resnet50(weights='DEFAULT')
    model3 = ModelFactory().get_model('resnet', 'imagenet', True)
    results = []
    for model in [model, model2, model3]:
        results.append(evaluate_model_texture_bias(model))
    print(results)
