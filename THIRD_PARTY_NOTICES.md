# Third-party notices

This repository contains project-authored software under the MIT License and
research data or metadata derived from third-party resources. The project
license does not replace the licenses, terms, or attribution requirements of
those resources.

## Massive-STEPS Tokyo data

- Project: **Massive-STEPS: Massive Semantic Trajectories for Understanding
  POI Check-ins — Dataset and Benchmarks**
- Authors: Wilson Wongso, Hao Xue, and Flora D. Salim
- Repository: <https://github.com/CRUISEResearchGroup/Massive-STEPS>
- Paper: <https://arxiv.org/abs/2505.11239>
- Upstream repository license: Apache License 2.0
- License text: <https://github.com/CRUISEResearchGroup/Massive-STEPS/blob/main/LICENSE>

The CSV files under `data/` originate from or are derived from the Tokyo
portion of Massive-STEPS. Cite the upstream dataset paper when using these
artifacts. Massive-STEPS itself incorporates upstream Semantic Trails and POI
metadata; downstream users remain responsible for reviewing the source
dataset documentation and any applicable terms.

## Multilingual MiniLM model

- Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Pinned revision: `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`
- Model card: <https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2>
- License: Apache License 2.0
- License text: <https://www.apache.org/licenses/LICENSE-2.0>

Model weights are not committed to this repository. They can be downloaded
and validated locally with `python main.py download-models`.

## Multilingual E5-small model

- Model: `intfloat/multilingual-e5-small`
- Pinned candidate revision: `614241f622f53c4eeff9890bdc4f31cfecc418b3`
- Model card: <https://huggingface.co/intfloat/multilingual-e5-small>
- License: MIT License
- License text: <https://opensource.org/license/mit>

E5 model weights are not committed to this repository. The P0-1/P0-2
candidate rerun downloads and validates them locally with
`python main.py download-models --model multilingual-e5-small`.

## Python dependencies

Python packages listed in `requirements.txt` are not vendored. Each package is
distributed under its own license. Review the package metadata and license
before redistribution, especially when building a binary distribution.
