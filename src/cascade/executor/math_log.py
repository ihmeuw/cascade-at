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

INSTANTANIOUSNESS_THRESHOLD = 0.5


class MathLogFormatter(Formatter):
    def __init__(self):
        datefmt = "%y-%m-%d %H:%M:%S"
        super().__init__(datefmt=datefmt)
        self.in_dismod_output = False
        self.last_event = 0.0
        self.has_emited_css = False

    def format(self, record):
        if not self.has_emited_css:
            message = _CSS
            self.has_emited_css = True
        else:
            message = ""

        is_dismod = record.__dict__.pop("is_dismod_output", False)
        if self.in_dismod_output and not is_dismod:
            message += "</div>"
            self.in_dismod_output = False
        elif not self.in_dismod_output and is_dismod:
            message += "<div class='dismod_block'>"
            self.in_dismod_output = True

        if not is_dismod:
            if record.created - self.last_event > INSTANTANIOUSNESS_THRESHOLD:
                message += f"<div class='time_stamp'>{self.formatTime(record, self.datefmt)}</div>"
            level_class = record.levelname.lower()
            message += f"<div class='log_line {level_class}'>"
            message += f"<span class='line_prefix'><span class='level_name {level_class}'>{record.levelname}</span> "
            message += f"<span class='function_name'>{record.funcName}</span></span>"
        self.last_event = record.created

        message += record.getMessage().replace("\n", "<br/>\n")
        if not is_dismod:
            message += "</div>"

        return message
