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
