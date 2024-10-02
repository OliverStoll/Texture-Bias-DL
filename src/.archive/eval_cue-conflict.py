import os
import json
from PIL import Image
from sanity_checks.check_transforms import to_tensor
from torchvision.models import resnet50

from models import ModelFactory
from utils.config import CONFIG
from utils.logger import create_logger
from probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping


class CueConflictEvaluation:
    log = create_logger("Cue-Conflict Evaluation")
    stimuli_path = '/evaluation/cue-conflict/texture-vs-shape/stimuli/style-transfer-preprocessed-512'
    categories = [
        "airplane",
        "bear",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "clock",
        "dog",
        "elephant",
        "keyboard",
        "knife",
        "oven",
        "truck"
    ]
    mapping = ImageNetProbabilitiesTo16ClassesMapping()  # function provided by the paper

    def evaluate_model_texture_bias(self, model, verbose=True):
        """ Own method, not part of the paper """
        # following the paper, we only count the Texture Bias by counting correct predictions
        # and then dividing
        correct_texture_preds = 0
        correct_shape_preds = 0
        tb_per_category = {category: [] for category in self.categories}
        tb_avg_per_category = {category: None for category in self.categories}
        for category in self.categories:
            category_path = f"{self.stimuli_path}/{category}"
            category_image_names = os.listdir(category_path)
            for category_image_name in category_image_names:
                texture_name_raw = category_image_name.split('-')[1].replace('.png', '')
                texture = ''.join([i for i in texture_name_raw if not i.isdigit()])
                if texture == category:  # skip if no cue-conflict
                    continue
                example_image = Image.open(f"{category_path}/{category_image_name}")
                example_tensor = to_tensor(example_image)
                example_tensor = example_tensor.unsqueeze(0)
                output = model(example_tensor)
                output = output.squeeze(0)
                softmax_output = output.softmax(dim=0).detach().numpy()
                output = output.detach().numpy()
                predicted_category = self.mapping.probabilities_to_decision(softmax_output)

                # check if predicted is either category or texture, else skip (according to the paper)
                if not (predicted_category == category or predicted_category == texture):
                    continue
                if predicted_category == category:
                    correct_shape_preds += 1
                elif predicted_category == texture:
                    correct_texture_preds += 1

                classes_softmax = self.mapping.softmax_of_16_classes(raw_probabilities=output)
                classes_softmax_dict = dict(zip(self.categories, classes_softmax))
                category_probability = classes_softmax_dict[category]
                texture_probability = classes_softmax_dict[texture]
                texture_bias = texture_probability / (category_probability + texture_probability)
                tb_per_category[category].append(texture_bias)
                if verbose:
                    self.log.debug(
                        f"[{correct_shape_preds:3d}|{correct_texture_preds:3d}]  "
                        f" | {texture_bias * 100:3.0f} %"
                        f" | {category:<8} ({category_probability:.2f})"
                        f" | {texture:<8} ({texture_probability:.2f})"
                    )
        # calculate average texture bias for each category
        for category in tb_per_category.keys():
            tb_avg_per_category[category] = sum(tb_per_category[category]) / len(tb_per_category[category])

        tb_avg_per_category['TEXTURE_BIAS'] = (correct_texture_preds / (correct_texture_preds + correct_shape_preds))
        tb_avg_per_category['SHAPE_BIAS'] = (correct_shape_preds / (correct_texture_preds + correct_shape_preds))
        self.log.info(f"Texture-Bias after Geirhos: {tb_avg_per_category['TEXTURE_BIAS'] * 100:.2f} %")
        return tb_avg_per_category

    def evaluate_multiple_models_texture_bias(self, model_names: list, verbose=True):
        all_outputs = {}
        for idx, model_name in enumerate(model_names):
            self.log.info(f"[{idx+1}/{len(model_names)}] Evaluating cue-conflict: {model_name}")
            model = ModelFactory().get_model(model_name, CONFIG['datasets']['imagenet'],
                                             pretrained=True)
            category_tb = self.evaluate_model_texture_bias(model, verbose)
            average_texture_bias = sum(category_tb.values()) / len(category_tb)
            category_tb['AVERAGE'] = average_texture_bias
            all_outputs[model_name] = category_tb
            # log results
            self.log.info(f"[{model_name}] Percent of texture-based decisions per category (only including ones where either texture or category was correctly predicted):")
            for category, tb in category_tb.items():
                self.log.info(f"\t{category:<8} -> {tb * 100:3.0f} %")

        return all_outputs


if __name__ == '__main__':
    model = resnet50(weights="DEFAULT")
    tb_results = CueConflictEvaluation().evaluate_model_texture_bias(model, verbose=True)
    exit()


    model_names = ModelCollection().all_model_names

    texture_bias_results = CueConflictEvaluation().evaluate_multiple_models_texture_bias(
        model_names=model_names,
        verbose=True,
    )
    # save results
    with open('../evaluation/texture-bias-results.json', 'w') as file:
        json.dump(texture_bias_results, file)
