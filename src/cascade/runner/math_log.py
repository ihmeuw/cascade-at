from logging import Formatter

_CSS = """<style>
.dismod_block {
    background: hsl(0, 0%, 90%);
    padding: 5px;
    margin: 10px;
}
.line_prefix {
    color: hsl(0, 0%, 50%);
    display: inline-block;
    width: 25em;
    border-right: black solid 1px;
    margin-right: 1em;
    padding-top: 3px;
    padding-bottom: 3px;
}
.log_line {
}
.time_stamp {
    font-weight: bold;
    padding: 10px;
}
.level_name {
    display: inline-block;
    width: 6em;
}
.level_name.info, .level_name.debug {
    color: hsl(0, 0%, 50%);
}
.level_name.warning {
    color: hsl(30, 100%, 45%);
}
.level_name.error {
    color: red;
}

.log_line.warning {
    background: hsl(52, 100%, 79%);
}

.log_line.error {
    background: hsl(0, 100%, 79%);
}

</style>
"""

INSTANTANEOUSNESS_THRESHOLD = 0.5
TIMESTAMP_SPACING = 60.0


class MathLogFormatter(Formatter):
    def __init__(self):
        datefmt = "%y-%m-%d %H:%M:%S"
        super().__init__(datefmt=datefmt)
        self.in_dismod_output = False
        self.last_event = 0.0
        self.last_timestamp = 0.0
        self.has_emitted_css = False

    def format(self, record):
        if not self.has_emitted_css:
            message = _CSS
            self.has_emitted_css = True
        else:
            message = ""

        assure_timestamp = record.__dict__.pop("assure_timestamp", False)

        is_dismod = record.__dict__.pop("is_dismod_output", False)
        if self.in_dismod_output and not is_dismod:
            message += "</pre>"
            self.in_dismod_output = False
        elif not self.in_dismod_output and is_dismod:
            message += "<pre class='dismod_block'>"
            self.in_dismod_output = True

        if not is_dismod:
            if assure_timestamp or \
               record.created - self.last_event > INSTANTANEOUSNESS_THRESHOLD or \
               record.created - self.last_timestamp > TIMESTAMP_SPACING:
                message += f"<div class='time_stamp'>{self.formatTime(record, self.datefmt)}</div>"
                self.last_timestamp = record.created
            level_class = record.levelname.lower()
            message += f"<div class='log_line {level_class}'>"
            message += f"<span class='line_prefix'><span class='level_name {level_class}'>{record.levelname}</span> "
            message += f"<span class='function_name'>{record.funcName}</span></span>"
            message += record.getMessage().replace("\n", "<br/>\n")
            if record.exc_info:
                message += "<pre class='stacktrace_block'>" + self.formatException(record.exc_info) + "</pre>"
        else:
            message += record.getMessage()
        self.last_event = record.created

        if not is_dismod:
            message += "</div>"

        return message
