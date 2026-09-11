#!/usr/bin/env bash
# Shared "copy VibeSim/ the way the runner sees it" helper.
#
# `build-codex-runner-image.sh` (build context) and `test-codex-runner-image.sh`
# (smoke workspace) must assemble byte-identical trees: the test exists to prove
# the image can build the same source the image was built from, and any
# divergence makes it prove nothing. That invariant used to live in a comment
# and a duplicated loop, and the duplication silently broke when
# `alignment/load_generator/req-frontend` became a Cargo path dependency.
#
# Note this deliberately differs from the *runtime* workspace copy
# (`backend/codex_runtime/workspace.py`), which skips submodules because
# `docker.py` bind-mounts them read-only into each conversation container
# instead. Neither of these scripts has a container to mount into — the image
# build runs `cargo build` at layer-build time — so they must copy instead.

# Submodules the runner image does not need. The alignment profilers are
# host-only nsys tooling with multi-GB working trees, and nothing in the
# container builds or imports them. Everything else is copied, so a newly added
# submodule fails safe: a larger image rather than a broken `cargo build`.
MAIN_TREE_SKIPPED_SUBMODULES=(
  "alignment/profiler/vllm"
  "alignment/profiler/sglang"
)

_main_tree_is_skipped_submodule() {
  local candidate="$1" skipped
  for skipped in "${MAIN_TREE_SKIPPED_SUBMODULES[@]}"; do
    [ "$candidate" = "$skipped" ] && return 0
  done
  return 1
}

# Copy the current contents of one repo's tracked files, preserving intentional
# working-tree edits while ignoring untracked build artifacts.
_main_tree_copy_tracked_files() {
  local source_root="$1" destination_root="$2" prefix="$3"
  local rel_path source_path listing
  listing="$(mktemp "${TMPDIR:-/tmp}/vibesim-main-files.XXXXXX")" || return 1
  if ! git -c core.fsmonitor=false -C "$source_root" ls-files -z > "$listing"; then
    rm -f "$listing"
    return 1
  fi
  while IFS= read -r -d '' rel_path; do
    source_path="$source_root/$rel_path"
    # `git ls-files` also reports gitlinks and index-only (deleted) paths, which
    # are neither regular files nor symlinks on disk.
    if [ -f "$source_path" ] || [ -L "$source_path" ]; then
      if ! mkdir -p "$destination_root/$prefix$(dirname "$rel_path")" \
        || ! cp -a "$source_path" "$destination_root/$prefix$rel_path"; then
        rm -f "$listing"
        return 1
      fi
    fi
  done < "$listing"
  rm -f "$listing"
}

# copy_main_tree <main_dir> <destination_root>
#
# Copies VibeSim/'s tracked files plus the tracked files of every submodule that is
# not in MAIN_TREE_SKIPPED_SUBMODULES. Fails loudly on an uninitialized
# submodule rather than producing a tree that cannot build.
copy_main_tree() {
  local main_dir="$1" destination_root="$2" submodule_path entry listing status

  mkdir -p "$destination_root" || return 1
  _main_tree_copy_tracked_files "$main_dir" "$destination_root" "" || return 1

  listing="$(mktemp "${TMPDIR:-/tmp}/vibesim-main-submodules.XXXXXX")" || return 1
  # config --null separates key and value by newline and records by NUL.
  # Missing files/no matching keys return 1; parse/read failures must propagate.
  if git -C "$main_dir" config --null -f .gitmodules \
    --get-regexp '^submodule\..*\.path$' > "$listing"; then
    status=0
  else
    status=$?
  fi
  if [ "$status" -ne 0 ] && [ "$status" -ne 1 ]; then
    rm -f "$listing"
    return "$status"
  fi
  while IFS= read -r -d '' entry; do
    submodule_path="${entry#*$'\n'}"
    [ -n "$submodule_path" ] || continue
    if _main_tree_is_skipped_submodule "$submodule_path"; then
      echo "skipping submodule (not needed in the runner image): $submodule_path" >&2
      continue
    fi
    if [ ! -e "$main_dir/$submodule_path/.git" ]; then
      echo "submodule is not initialized: $submodule_path" >&2
      echo "run: git -C $main_dir submodule update --init $submodule_path" >&2
      rm -f "$listing"
      return 1
    fi
    if ! _main_tree_copy_tracked_files \
      "$main_dir/$submodule_path" "$destination_root" "$submodule_path/"; then
      rm -f "$listing"
      return 1
    fi
  done < "$listing"
  rm -f "$listing"
}
