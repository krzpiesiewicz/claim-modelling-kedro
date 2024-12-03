from colored import fg, attr
import colorlog
import logging
import textwrap


class IndentedMessageColoredFormatter(colorlog.ColoredFormatter):
    def __init__(self, *args, **kwargs):
        self.custom_log_colors = kwargs.pop("custom_log_colors", {})
        self.indent_size = kwargs.pop("indent_size", 0)
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        lines = super().formatMessage(record).split("\n")
        for i in range(1, len(lines)):
            lines[i] = " " * self.indent_size + lines[i]
        message = '\n'.join(lines)

        if record.levelname in self.custom_log_colors:
            color = self.custom_log_colors[record.levelname]
            message = f"{fg(color)}{message}{attr('reset')}"
        return message


class IndentedMessageFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self.indent_size = kwargs.pop("indent_size", 0)
        self.wrap_width = kwargs.pop("wrap_width", None)
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        message = super().formatMessage(record)

        if self.wrap_width is not None:
            lines = []
            for line in message.split("\n"):
                lines += textwrap.wrap(line, width=self.wrap_width, break_long_words=False, break_on_hyphens=False)
        else:
            lines = message.split("\n")

        for i in range(len(lines)):
            if i != 0:
                lines[i] = " " * self.indent_size + lines[i]

        message = '\n'.join(lines)
        return message
