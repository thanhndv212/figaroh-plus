# Changelog fragments

Each user-visible change adds one file here instead of editing
`CHANGELOG.md`. Parallel pull requests then touch different files and do not
conflict. The release pull request runs `towncrier build`, which writes the
dated section into `CHANGELOG.md` and deletes the fragments.

## Naming

`<issue>.<type>.md`, where `<type>` is one of `added`, `changed`,
`deprecated`, `removed`, `fixed`, `security` (the Keep a Changelog headings).

- `274.changed.md`: one entry for issue #274.
- `274.fixed.1.md`: a second entry of the same type for the same issue.
- `+short-slug.fixed.md`: an entry with no issue (the `+` prefix).

The issue number in the name only keeps names unique; it is not added to the
text. Write the entry as before, with references in the prose, for example
`(#264, plan item Q2.1)`.

## Content

One bullet's text, without the leading `- `, wrapped at about 78 columns.
Do not indent continuation lines; `towncrier` indents them. One paragraph per
fragment; split a change with two separate effects into two fragments.

## Commands (in `figaroh-dev`)

```bash
towncrier create 274.changed.md --edit   # or write the file by hand
towncrier build --draft --version 1.3.0  # preview the release section
```

`template.md.jinja` renders the section; `pyproject.toml` `[tool.towncrier]`
holds the configuration.
