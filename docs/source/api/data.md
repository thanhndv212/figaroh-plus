# Data contract

Typed inputs that carry their conventions: joint order, clock, signal origin,
effort kind and unit, valid-sample mask, session and source files. See the
decision record
[Data and result contract](https://github.com/thanhndv212/figaroh-plus/blob/devel/docs/decisions/data-result-contract.md).
Each type converts to and from the legacy dictionaries and CSV loader.

::: figaroh.data.trajectory
    options:
      show_root_heading: false

::: figaroh.data.observations
    options:
      show_root_heading: false

::: figaroh.data.source
    options:
      show_root_heading: false

::: figaroh.data.protocol
    options:
      show_root_heading: false
