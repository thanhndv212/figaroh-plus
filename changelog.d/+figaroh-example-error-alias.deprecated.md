`FigarohExampleError` (in `figaroh.utils` and
`figaroh.utils.error_handling`) is an alias of `FigarohError` that emits a
`DeprecationWarning` when it is looked up, and goes away in 2.0. `except`
clauses on the old name keep catching the library's errors meanwhile.
