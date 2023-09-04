from django import template
from django.utils.safestring import mark_safe

from base.indicators import get_indicators

INDICATORS = get_indicators()

register = template.Library()


@register.simple_tag(takes_context=True)
def symbol_render_html_snippet(context, symbol):
    return mark_safe(symbol.render_html_snippet())


@register.simple_tag(takes_context=True)
def symbol_indicators(context, symbol):
    rendered = ""
    for indicator in INDICATORS.values():
        rendered += template.loader.render_to_string(
            indicator.template, {"symbol": symbol}
        )
    return mark_safe(rendered)


@register.simple_tag(takes_context=True)
def indicators_radio_buttons(context):
    rendered = ""
    for indicator in INDICATORS.values():
        rendered += template.loader.render_to_string(
            "base/indicators/indicator_radio_button.html",
            {"js_slug": indicator.js_slug},
        )
    return mark_safe(rendered)


@register.simple_tag(takes_context=True)
def indicators_sorting_js(context):
    rendered = ""
    for indicator in INDICATORS.values():
        rendered += template.loader.render_to_string(
            indicator.js_sorting,
            {},
        )
    return mark_safe(rendered)


@register.simple_tag(takes_context=True)
def bot_render_html_snippet(context, bot):
    return mark_safe(bot.render_html_snippet())


@register.filter("timedelta", is_safe=True)
def timedelta(value):
    if not value:  # pragma: no cover
        return ""
    secs = value.total_seconds()
    hours, rem = int(secs // 3600), int(secs % 3600)
    mins, rem = int(rem // 60), int(rem % 60)
    secs = rem
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


@register.filter("get", is_safe=True)
def get(value, key):
    if not value:  # pragma: no cover
        return ""
    return value.get(key, "")


@register.inclusion_tag("base/_action_button.html", takes_context=True)
def action_button(context, action, text, css_classes):
    return {
        "action": action,
        "text": text,
        "css_classes": css_classes,
    }


@register.inclusion_tag("base/_groups_breadcrumbs.html", takes_context=True)
def groups_breadcrumbs(context, user):
    context["groups"] = user.botgroups.all().order_by("name")
    return context
