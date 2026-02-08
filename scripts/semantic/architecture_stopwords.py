"""Software/architecture-level stopwords.

Goal: reduce 'framework hallucination' in semantic similarity by removing tokens that
primarily indicate technical implementation rather than domain meaning.

This list is intentionally aggressive; tune per system if needed.
"""

# Generic Java/framework tokens
ARCH_STOPWORDS = {
    # common architecture / layering
    "impl", "implementation", "service", "services", "rest", "controller", "resource",
    "endpoint", "api", "client", "server", "provider", "consumer",
    "handler", "dispatcher", "filter", "interceptor", "middleware",
    "component", "bean", "configuration", "config", "context",
    "factory", "builder", "manager", "helper", "util", "utils",
    "adapter", "converter", "mapper", "wrapper", "proxy", "delegate",
    "repository", "dao", "entity", "entities", "model", "domain", "dto", "vo", "po",
    "request", "response", "payload", "params", "param", "argument",
    "action", "form", "page", "view", "html", "jsp", "jsf",
    # web tech
    "http", "https", "uri", "url", "json", "xml", "yaml",
    "get", "set", "post", "put", "delete", "patch",
    # persistence
    "jpa", "hibernate", "jdbc", "orm", "sql", "nosql", "mongo",
    "table", "column", "row", "schema",
    # common noise
    "data", "value", "values", "type", "types", "name", "names", "id", "ids",
    "base", "default", "common", "core",
}

# ------------------------------
# Domain-Focus Filter (hard list)
# ------------------------------
# These are *intentionally* aggressive, architecture/implementation-oriented tokens that
# frequently appear across many classes in enterprise Java systems but carry little
# business-domain signal. They are used to purify semantic text features before embedding.
#
# Notes:
# - Keep tokens lowercase.
# - Place only general technical tokens here; project-specific tokens are injected elsewhere.
# - This list is designed to be auditable and paper-friendly.
DOMAIN_FOCUS_HARD_STOPWORDS = {
    # common layers/components
    "impl", "implementation", "service", "services", "svc",
    "controller", "resource", "endpoint", "api",
    "dao", "repository", "repo",
    "manager", "factory", "provider", "helper", "handler", "processor",
    "util", "utils", "utility", "common", "shared", "base", "abstract",
    "config", "configuration", "properties", "property",
    "client", "server", "gateway", "proxy", "adapter",
    "listener", "filter", "interceptor", "middleware",

    # persistence / data
    "db", "database", "datasource", "jdbc", "jpa", "hibernate", "orm",
    "entity", "entities", "model", "domain", "dto", "vo", "po", "bean",
    "schema", "table", "column", "row",

    # web/container/framework noise
    "servlet", "jsp", "jsf", "spring", "springboot", "jakarta", "javaee",
    "ejb", "jms", "rest", "soap", "http", "https",

    # testing/logging/ops
    "test", "tests", "testing", "mock", "stub",
    "log", "logger", "logging", "trace", "tracing", "metric", "metrics",
}


def get_all_arch_stopwords(*, extra: list[str] | None = None, include_domain_focus_hard: bool = True) -> set[str]:
    """Return merged architecture stopwords.

    This keeps a single canonical place for the hard-lists so Phase1 semantic purification
    remains reproducible across scripts.
    """
    out = set(ARCH_STOPWORDS)
    if include_domain_focus_hard:
        out |= set(DOMAIN_FOCUS_HARD_STOPWORDS)
    if extra:
        out |= {str(x).strip().lower() for x in extra if str(x).strip()}
    return out
