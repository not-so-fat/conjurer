import altair


LIGHT_GRAY = '#d3d3d3'
DARK_GRAY = '#a9a9a9'


def conjurer():
    return dict(
        config=dict(
            axis=dict(
                grid=True,
                gridColor=LIGHT_GRAY,
                tickColor=LIGHT_GRAY,
                domainColor=LIGHT_GRAY,
                labelColor=DARK_GRAY,
                titleColor=DARK_GRAY
            ),
            legend=dict(
                orient='top',
                titleAnchor='start',
                columns=0,
                titleColor=DARK_GRAY,
                labelColor=DARK_GRAY
            ),
            title=dict(
                align='left',
                anchor='start'
            )
        ),
    )


def apply_theme():
    altair.themes.register('conjurer', conjurer)
    altair.themes.enable('conjurer')
