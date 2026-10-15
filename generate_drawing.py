import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_technical_drawing():
    fig, ax = plt.subplots(figsize=(11.69, 8.27), dpi=300) # A4 landscape

    # Page bounds in mm coords
    # Drawing border: 20mm left margin, 5mm top/bottom/right margin according to GOST
    border_left = 20
    border_right = 292
    border_bottom = 5
    border_top = 205

    # Outer frame setup
    ax.set_xlim(0, 297)
    ax.set_ylim(0, 210)
    ax.set_aspect('equal')
    ax.axis('off')

    # Drawing frame according to GOST (20mm left, 5mm top/bottom/right)
    frame = patches.Rectangle((border_left, border_bottom), border_right - border_left, border_top - border_bottom,
                              linewidth=1.5, edgecolor='black', facecolor='none')
    ax.add_patch(frame)

    # Title Block (Основная надпись по ГОСТ 2.104-68, Форма 1)
    tb_w = 185
    tb_h = 55
    tb_x = border_right - tb_w  # 292 - 185 = 107
    tb_y = border_bottom        # 5

    tb_border = patches.Rectangle((tb_x, tb_y), tb_w, tb_h, linewidth=1.2, edgecolor='black', facecolor='none')
    ax.add_patch(tb_border)

    # Horizontal grid lines in Title Block
    y_lines = [tb_y + 5, tb_y + 10, tb_y + 15, tb_y + 20, tb_y + 25, tb_y + 35, tb_y + 40, tb_y + 45, tb_y + 50]
    for yl in y_lines:
        ax.plot([tb_x, tb_x + 65], [yl, yl], color='black', linewidth=0.7)

    # Vertical grid lines in Title Block
    ax.plot([tb_x + 7, tb_x + 7], [tb_y, tb_y + 25], color='black', linewidth=0.7)
    ax.plot([tb_x + 17, tb_x + 17], [tb_y, tb_y + 25], color='black', linewidth=0.7)
    ax.plot([tb_x + 27, tb_x + 27], [tb_y, tb_y + 25], color='black', linewidth=0.7)
    ax.plot([tb_x + 42, tb_x + 42], [tb_y, tb_y + 25], color='black', linewidth=0.7)
    ax.plot([tb_x + 57, tb_x + 57], [tb_y, tb_y + 25], color='black', linewidth=0.7)
    ax.plot([tb_x + 65, tb_x + 65], [tb_y, tb_y + 55], color='black', linewidth=1.2)

    # Upper/Right sections of title block
    ax.plot([tb_x + 65, border_right], [tb_y + 15, tb_y + 15], color='black', linewidth=0.7)
    ax.plot([tb_x + 65, border_right], [tb_y + 20, tb_y + 20], color='black', linewidth=0.7)
    ax.plot([tb_x + 65, border_right], [tb_y + 35, tb_y + 35], color='black', linewidth=1.2)

    ax.plot([tb_x + 135, tb_x + 135], [tb_y + 15, tb_y + 35], color='black', linewidth=0.7)
    ax.plot([tb_x + 150, tb_x + 150], [tb_y + 15, tb_y + 35], color='black', linewidth=0.7)
    ax.plot([tb_x + 167, tb_x + 167], [tb_y + 15, tb_y + 35], color='black', linewidth=0.7)

    # Text in Title Block
    ax.text(tb_x + 100, tb_y + 42, 'Колесо крановое двухребордное', fontsize=10, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')
    ax.text(tb_x + 100, tb_y + 48, 'КК.300.90.00.000 СБ', fontsize=9, fontweight='bold', ha='center', va='center', fontfamily='sans-serif')
    ax.text(tb_x + 100, tb_y + 25, 'Сталь 65Г ГОСТ 14959-2016', fontsize=8, ha='center', va='center', fontfamily='sans-serif')

    # Labels in table
    ax.text(tb_x + 2, tb_y + 22, 'Изм.', fontsize=5, fontfamily='sans-serif')
    ax.text(tb_x + 9, tb_y + 22, 'Лист', fontsize=5, fontfamily='sans-serif')
    ax.text(tb_x + 18, tb_y + 22, '№ докум.', fontsize=5, fontfamily='sans-serif')
    ax.text(tb_x + 44, tb_y + 22, 'Подп.', fontsize=5, fontfamily='sans-serif')
    ax.text(tb_x + 58, tb_y + 22, 'Дата', fontsize=5, fontfamily='sans-serif')

    ax.text(tb_x + 2, tb_y + 17, 'Разраб.', fontsize=6, fontfamily='sans-serif')
    ax.text(tb_x + 2, tb_y + 12, 'Пров.', fontsize=6, fontfamily='sans-serif')
    ax.text(tb_x + 2, tb_y + 2, 'Н.контр.', fontsize=6, fontfamily='sans-serif')
    ax.text(tb_x + 2, tb_y + 7, 'Утв.', fontsize=6, fontfamily='sans-serif')

    ax.text(tb_x + 20, tb_y + 17, 'Инженер', fontsize=6, fontfamily='sans-serif')
    ax.text(tb_x + 20, tb_y + 12, 'Глав.спец', fontsize=6, fontfamily='sans-serif')

    ax.text(tb_x + 138, tb_y + 30, 'Лит.', fontsize=6, fontfamily='sans-serif')
    ax.text(tb_x + 152, tb_y + 30, 'Масса', fontsize=6, fontfamily='sans-serif')
    ax.text(tb_x + 170, tb_y + 30, 'Масштаб', fontsize=6, fontfamily='sans-serif')
    ax.text(tb_x + 143, tb_y + 20, 'У', fontsize=7, fontweight='bold', fontfamily='sans-serif')
    ax.text(tb_x + 158, tb_y + 20, '35 кг', fontsize=7, fontfamily='sans-serif')
    ax.text(tb_x + 176, tb_y + 20, '1:2.5', fontsize=7, fontfamily='sans-serif')

    ax.text(tb_x + 100, tb_y + 8, 'Завод тяжелого машиностроения', fontsize=8, ha='center', va='center', fontfamily='sans-serif')

    # Technical requirements (Top Left corner)
    tech_reqs = [
        "1. *Размеры для справок.",
        "2. HRC 32...38 поверхности катания и реборд.",
        "3. Шпоночный паз по ГОСТ 23360-78 (b=14 H9, t2=3.8+0.2 мм).",
        "4. Предельные отклонения размеров: H14, h14, ±IT14/2.",
        "5. Покрытие: Грунтовка ГФ-021 ГОСТ 25129-82."
    ]
    req_x = border_left + 5
    req_y = border_top - 8
    ax.text(req_x, req_y, "Технические требования:", fontsize=8, fontweight='bold', fontfamily='sans-serif')
    for i, req in enumerate(tech_reqs):
        ax.text(req_x, req_y - 4.5 - i * 4.0, req, fontsize=7, fontfamily='sans-serif', va='top')

    # --- VIEWS GEOMETRY ---
    scale = 0.35

    # VIEW 1: Cross-Section View (Frontal Section)
    cx1 = 65
    cy1 = 95

    w_tot = 90 * scale
    w_flange = 15 * scale
    w_tread = 60 * scale
    w_web = 30 * scale

    r_flange = 170 * scale
    r_tread = 150 * scale
    r_rim_in = 120 * scale
    r_hub = 55 * scale
    r_bore = 25 * scale

    x_left = cx1 - w_tot / 2.0
    x_right = cx1 + w_tot / 2.0

    # Upper Half Section (Y > cy1)
    upper_poly_top = [
        (x_left, cy1 + r_flange),
        (x_left + w_flange, cy1 + r_flange),
        (x_left + w_flange, cy1 + r_tread),
        (x_right - w_flange, cy1 + r_tread),
        (x_right - w_flange, cy1 + r_flange),
        (x_right, cy1 + r_flange),
        (x_right, cy1 + r_rim_in),
        (cx1 + w_web/2, cy1 + r_rim_in),
        (cx1 + w_web/2, cy1 + r_hub),
        (x_right, cy1 + r_hub),
        (x_right, cy1 + r_bore),
        (x_left, cy1 + r_bore),
        (x_left, cy1 + r_hub),
        (cx1 - w_web/2, cy1 + r_hub),
        (cx1 - w_web/2, cy1 + r_rim_in),
        (x_left, cy1 + r_rim_in),
        (x_left, cy1 + r_flange)
    ]

    xs_u, ys_u = zip(*upper_poly_top)
    ax.plot(xs_u, ys_u, color='black', linewidth=1.2)

    polygon_u = patches.Polygon(upper_poly_top, closed=True, facecolor='none', edgecolor='black', hatch='///', linewidth=1.2)
    ax.add_patch(polygon_u)

    # Lower Half Section (Y < cy1)
    lower_poly_bot = [
        (x_left, cy1 - r_flange),
        (x_left + w_flange, cy1 - r_flange),
        (x_left + w_flange, cy1 - r_tread),
        (x_right - w_flange, cy1 - r_tread),
        (x_right - w_flange, cy1 - r_flange),
        (x_right, cy1 - r_flange),
        (x_right, cy1 - r_rim_in),
        (cx1 + w_web/2, cy1 - r_rim_in),
        (cx1 + w_web/2, cy1 - r_hub),
        (x_right, cy1 - r_hub),
        (x_right, cy1 - (r_bore + 3.8 * scale)),
        (x_left, cy1 - (r_bore + 3.8 * scale)),
        (x_left, cy1 - r_hub),
        (cx1 - w_web/2, cy1 - r_hub),
        (cx1 - w_web/2, cy1 - r_rim_in),
        (x_left, cy1 - r_rim_in),
        (x_left, cy1 - r_flange)
    ]

    xs_l, ys_l = zip(*lower_poly_bot)
    ax.plot(xs_l, ys_l, color='black', linewidth=1.2)

    polygon_l = patches.Polygon(lower_poly_bot, closed=True, facecolor='none', edgecolor='black', hatch='///', linewidth=1.2)
    ax.add_patch(polygon_l)

    # Centerline for View 1
    ax.plot([x_left - 10, x_right + 10], [cy1, cy1], color='black', linestyle='-.', linewidth=0.6)

    # Inner transition lines for section view
    ax.plot([x_left + w_flange, x_left + w_flange], [cy1 + r_tread, cy1 - r_tread], color='black', linewidth=0.5, linestyle=':')
    ax.plot([x_right - w_flange, x_right - w_flange], [cy1 + r_tread, cy1 - r_tread], color='black', linewidth=0.5, linestyle=':')

    # --- DIMENSIONS FOR VIEW 1 ---
    def draw_dimension_h(x1, x2, y, text, offset_text=2):
        ax.annotate('', xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle='<->', lw=0.8, color='black'))
        ax.plot([x1, x1], [y - 2, y + 2], color='black', lw=0.5)
        ax.plot([x2, x2], [y - 2, y + 2], color='black', lw=0.5)
        ax.text((x1 + x2)/2.0, y + offset_text, text, fontsize=7.5, ha='center', va='bottom', fontfamily='sans-serif')

    def draw_dimension_v(y1, y2, x, text, offset_text=2):
        ax.annotate('', xy=(x, y1), xytext=(x, y2),
                    arrowprops=dict(arrowstyle='<->', lw=0.8, color='black'))
        ax.plot([x - 2, x + 2], [y1, y1], color='black', lw=0.5)
        ax.plot([x - 2, x + 2], [y2, y2], color='black', lw=0.5)
        ax.text(x - offset_text, (y1 + y2)/2.0, text, fontsize=7.5, ha='right', va='center', rotation=90, fontfamily='sans-serif')

    # Width dimensions below view 1
    y_dim_w1 = cy1 - r_flange - 10
    draw_dimension_h(x_left, x_right, y_dim_w1, "90")

    y_dim_w2 = cy1 - r_flange - 18
    draw_dimension_h(x_left + w_flange, x_right - w_flange, y_dim_w2, "60")

    # Extension lines for width
    ax.plot([x_left, x_left], [cy1 - r_flange, y_dim_w1 - 3], color='black', lw=0.5)
    ax.plot([x_right, x_right], [cy1 - r_flange, y_dim_w1 - 3], color='black', lw=0.5)
    ax.plot([x_left + w_flange, x_left + w_flange], [cy1 - r_tread, y_dim_w2 - 3], color='black', lw=0.5)
    ax.plot([x_right - w_flange, x_right - w_flange], [cy1 - r_tread, y_dim_w2 - 3], color='black', lw=0.5)

    # Diameter dimensions left of view 1
    x_dim_d1 = x_left - 10
    draw_dimension_v(cy1 - r_flange, cy1 + r_flange, x_dim_d1, "Ø 340")
    ax.plot([x_left, x_dim_d1 - 3], [cy1 + r_flange, cy1 + r_flange], color='black', lw=0.5)
    ax.plot([x_left, x_dim_d1 - 3], [cy1 - r_flange, cy1 - r_flange], color='black', lw=0.5)

    x_dim_d2 = x_left - 20
    draw_dimension_v(cy1 - r_tread, cy1 + r_tread, x_dim_d2, "Ø 300 h9")
    ax.plot([x_left + w_flange, x_dim_d2 - 3], [cy1 + r_tread, cy1 + r_tread], color='black', lw=0.5)
    ax.plot([x_left + w_flange, x_dim_d2 - 3], [cy1 - r_tread, cy1 - r_tread], color='black', lw=0.5)


    # VIEW 2: Side Projection / Frontal View of Wheel with Bore and Keyway
    cx2 = 200
    cy2 = 130

    circle_flange = patches.Circle((cx2, cy2), r_flange, edgecolor='black', facecolor='none', linewidth=1.2)
    circle_tread = patches.Circle((cx2, cy2), r_tread, edgecolor='black', facecolor='none', linewidth=0.7, linestyle='--')
    circle_hub = patches.Circle((cx2, cy2), r_hub, edgecolor='black', facecolor='none', linewidth=1.2)
    circle_bore = patches.Circle((cx2, cy2), r_bore, edgecolor='black', facecolor='none', linewidth=1.2)

    ax.add_patch(circle_flange)
    ax.add_patch(circle_tread)
    ax.add_patch(circle_hub)
    ax.add_patch(circle_bore)

    # Keyway GOST 23360-78: b = 14 mm, t2 = 3.8 mm
    b_kw = 14 * scale
    t2_kw = 3.8 * scale

    kw_rect = patches.Rectangle((cx2 - b_kw/2.0, cy2 + r_bore), b_kw, t2_kw,
                                edgecolor='black', facecolor='white', linewidth=1.2, zorder=3)
    ax.add_patch(kw_rect)

    # Centerlines for View 2 (stay above title block: y >= 65 mm)
    ax.plot([cx2 - r_flange - 8, cx2 + r_flange + 8], [cy2, cy2], color='black', linestyle='-.', linewidth=0.6)
    ax.plot([cx2, cx2], [cy2 - r_flange - 5, cy2 + r_flange + 8], color='black', linestyle='-.', linewidth=0.6)

    # --- DIMENSIONS FOR VIEW 2 / KEYWAY ---
    # Shaft bore diameter dimension
    ax.annotate('Ø 50 H7', xy=(cx2 - r_bore*0.707, cy2 - r_bore*0.707),
                xytext=(cx2 - r_flange - 10, cy2 - r_flange/2),
                arrowprops=dict(arrowstyle='->', lw=0.8, color='black'),
                fontsize=8, fontweight='bold', fontfamily='sans-serif')

    # Keyway width b = 14
    y_kw_dim = cy2 + r_bore + t2_kw + 8
    draw_dimension_h(cx2 - b_kw/2.0, cx2 + b_kw/2.0, y_kw_dim, "14 H9")
    ax.plot([cx2 - b_kw/2.0, cx2 - b_kw/2.0], [cy2 + r_bore + t2_kw, y_kw_dim + 2], color='black', lw=0.5)
    ax.plot([cx2 + b_kw/2.0, cx2 + b_kw/2.0], [cy2 + r_bore + t2_kw, y_kw_dim + 2], color='black', lw=0.5)

    # Keyway total height = 53.8 mm
    x_kw_h = cx2 + r_hub + 8
    draw_dimension_v(cy2 - r_bore, cy2 + r_bore + t2_kw, x_kw_h, "53.8 +0.2", offset_text=-2)
    ax.plot([cx2, x_kw_h + 3], [cy2 - r_bore, cy2 - r_bore], color='black', lw=0.5)
    ax.plot([cx2 + b_kw/2.0, x_kw_h + 3], [cy2 + r_bore + t2_kw, cy2 + r_bore + t2_kw], color='black', lw=0.5)

    # Section A-A view callout text above View 1
    ax.text(cx1, cy1 + r_flange + 8, "А - А", fontsize=11, fontweight='bold', ha='center', fontfamily='sans-serif')

    # Section line indicator on View 2
    ax.plot([cx2, cx2], [cy2 + r_flange + 2, cy2 + r_flange + 8], color='black', linewidth=2.0)
    ax.plot([cx2, cx2], [cy2 - r_flange - 2, cy2 - r_flange - 6], color='black', linewidth=2.0)
    ax.text(cx2 + 3, cy2 + r_flange + 6, "А", fontsize=10, fontweight='bold', fontfamily='sans-serif')
    ax.text(cx2 + 3, cy2 - r_flange - 7, "А", fontsize=10, fontweight='bold', fontfamily='sans-serif')

    plt.tight_layout()
    plt.savefig("crane_wheel_drawing.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.savefig("crane_wheel_drawing.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print("Technical drawing successfully regenerated: crane_wheel_drawing.pdf")

if __name__ == "__main__":
    create_technical_drawing()
