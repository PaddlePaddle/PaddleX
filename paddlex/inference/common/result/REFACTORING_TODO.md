# Refactoring TODO

Pending improvements tracked during the DocumentConverter extraction.

1. **Rewrite Word/LaTeX conversion** — fill in `WordConverter` / `LatexConverter` with
   proper implementations; retire `_to_word()` / `_to_latex()` in Result classes.
2. **Slim down mixin.py** — once Converters own the rendering logic, `WordMixin` /
   `LatexMixin` can delegate to them.
3. **Eliminate the dual-mode signature hack in `_to_word()`** — currently it inspects
   its own arguments to decide between two code paths.
4. **Unify Block data model** — define a `BaseDocumentBlock` Protocol so `LayoutBlock`
   and `PaddleOCRVLBlock` share a typed contract.
5. **Add `WordWriter` / `LatexWriter`** to `io/writers.py` alongside `MarkdownWriter`.
6. **Eliminate save_to_xxx file-type detection boilerplate** — the repeated
   `isinstance` / suffix checks in every `save_to_*` method.
7. **Unify `components/utils/mixin.py` with `common/result/mixin.py`** — two parallel
   Mixin hierarchies serve overlapping purposes.
8. **Refactor pp_doctranslation** — its markdown-based Word/LaTeX path should use the
   new Converter layer instead of its own inline conversion.
