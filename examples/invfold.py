import torch
import torch.nn.functional as F

from esm.pretrained import (
    ESM3_sm_open_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization.sequence_tokenizer import (
    EsmSequenceTokenizer,
)
from esm.utils.structure.protein_chain import ProteinChain

if __name__ == "__main__":
    tokenizer = EsmSequenceTokenizer()
    encoder = ESM3_structure_encoder_v0("cpu")
    model = ESM3_sm_open_v0("cpu")

    chain = ProteinChain.from_pdb("/Users/mukulsherekar/pythonProject/ESM/esm/data/1utn.pdb")
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords = coords.cpu()
    plddt = plddt.cpu()
    residue_index = residue_index.cpu()
    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)

    # Add BOS/EOS padding
    coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf)
    plddt = F.pad(plddt, (1, 1), value=0)
    structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    structure_tokens[:, 0] = 4098
    structure_tokens[:, -1] = 4097

    output = model.forward(
        structure_coords=coords, per_res_plddt=plddt, structure_tokens=structure_tokens
    )
    sequence_tokens = torch.argmax(output.sequence_logits, dim=-1)
    sequence = tokenizer.decode(sequence_tokens[0])
    print(sequence)
