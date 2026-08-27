"""Machine- and volume-scoped storage operations (portable storage roots, `PSR`).

Distinct from `smftools.datasets`, which bundles example data shipped with the
package. This subpackage covers a user's own storage: named roots and, from
`PSR-08` onward, volume identity for drives that move between machines. See
`dev/plans/in-progress/portable_storage_roots_implementation_plan.md`.
"""

from __future__ import annotations
