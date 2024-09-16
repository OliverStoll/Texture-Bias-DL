from torch import nn
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.models import utils


class MAE(nn.Module):
    def __init__(self, vit, decoder_dim=512, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = vit.patch_embed.patch_size[0]

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def forward(self, images):
        batch_size = images.shape[0]

        # Generate mask indices
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.backbone.encode(images=images, idx_keep=idx_keep)

        # Decoder input preparation & decode
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))
        x_decoded = self.decoder.decode(x_masked)

        # Predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)

        # Get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        target = utils.get_at_index(patches, idx_mask - 1)

        # Reconstruct the full image by combining kept and predicted patches
        patches_reconstructed = utils.set_at_index(patches, idx_mask - 1, x_pred)
        images_reconstructed = utils.unpatchify(patches_reconstructed, self.patch_size)

        return x_pred, target, images_reconstructed