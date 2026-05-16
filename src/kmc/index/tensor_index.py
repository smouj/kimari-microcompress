"""Tensor index: maps tensor names to their metadata and block lists.

TensorIndex provides fast lookup of tensor metadata by name, enabling
selective extraction of specific tensors. This index is only available
for archives that include tensor-aware metadata (created with
--tensor-aware mode). For older archives without tensor metadata,
tensor-level access is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TensorLocation:
    """Location and metadata for a single tensor within the archive.

    Attributes:
        name: Name of the tensor (e.g., 'model.layers.0.mlp.down_proj.weight').
        file_path: Relative path of the file containing this tensor.
        dtype: Data type of the tensor (e.g., 'BF16', 'FP16', 'FP32').
            None if tensor metadata is not available.
        shape: Shape of the tensor as a list of integers.
            None if tensor metadata is not available.
        block_ids: Ordered list of block IDs that compose this tensor's data.
    """

    name: str
    file_path: str
    dtype: str | None
    shape: list[int] | None
    block_ids: list[int] = field(default_factory=list)


class TensorIndex:
    """Index of all tensors in a .kmc archive, optimized for partial access.

    Supports lookup by tensor name and provides the block IDs needed
    to extract a specific tensor. This index is only populated for
    archives created with --tensor-aware mode.
    """

    def __init__(self) -> None:
        self._tensors: dict[str, TensorLocation] = {}
        self._ordered: list[TensorLocation] = []

    def add(self, tensor_loc: TensorLocation) -> None:
        """Add a tensor location to the index."""
        self._tensors[tensor_loc.name] = tensor_loc
        self._ordered.append(tensor_loc)

    def get(self, name: str) -> TensorLocation | None:
        """Look up a tensor by its name."""
        return self._tensors.get(name)

    def list_tensors(self) -> list[str]:
        """List all tensor names in the archive."""
        return [t.name for t in self._ordered]

    @property
    def total_tensors(self) -> int:
        """Total number of indexed tensors."""
        return len(self._tensors)

    @property
    def available(self) -> bool:
        """Whether tensor-level access is available.

        Returns True if the archive contains tensor metadata.
        """
        return len(self._tensors) > 0

    @classmethod
    def from_manifest(cls, manifest: object) -> TensorIndex:
        """Build a TensorIndex from a KMCManifest.

        Tensors are extracted from:
        1. File-level tensor_entries (safetensors metadata)
        2. Block-level tensor_name/tensor_dtype/tensor_shape fields

        Args:
            manifest: KMCManifest instance.

        Returns:
            Populated TensorIndex.
        """
        index = cls()
        global_block_id = 0

        for file_entry in manifest.files:  # type: ignore[attr-defined]
            # Build a map of tensor_name -> block_ids from blocks
            tensor_blocks: dict[str, list[int]] = {}
            tensor_dtypes: dict[str, str] = {}
            tensor_shapes: dict[str, list[int]] = {}

            for block in file_entry.blocks:  # type: ignore[attr-defined]
                tname = block.tensor_name  # type: ignore[attr-defined]
                if tname:
                    if tname not in tensor_blocks:
                        tensor_blocks[tname] = []
                    tensor_blocks[tname].append(global_block_id)

                    if block.tensor_dtype:  # type: ignore[attr-defined]
                        tensor_dtypes[tname] = block.tensor_dtype  # type: ignore[attr-defined]
                    if block.tensor_shape:  # type: ignore[attr-defined]
                        tensor_shapes[tname] = block.tensor_shape  # type: ignore[attr-defined]

                global_block_id += 1

            # Also get tensor info from file-level tensor_entries
            tensor_entry_map: dict[str, dict] = {}
            for te in file_entry.tensor_entries:  # type: ignore[attr-defined]
                tensor_entry_map[te.name] = {  # type: ignore[attr-defined]
                    "dtype": te.dtype,  # type: ignore[attr-defined]
                    "shape": te.shape,  # type: ignore[attr-defined]
                }

            # Merge: prefer file-level tensor_entries for dtype/shape,
            # use block-level for block_ids
            all_tensor_names = set(tensor_blocks.keys()) | set(tensor_entry_map.keys())

            for name in sorted(all_tensor_names):
                entry_info = tensor_entry_map.get(name, {})
                block_ids = tensor_blocks.get(name, [])

                dtype = entry_info.get("dtype") or tensor_dtypes.get(name)
                shape = entry_info.get("shape") or tensor_shapes.get(name)

                loc = TensorLocation(
                    name=name,
                    file_path=file_entry.path,  # type: ignore[attr-defined]
                    dtype=dtype,
                    shape=shape,
                    block_ids=block_ids,
                )
                index.add(loc)

        return index
