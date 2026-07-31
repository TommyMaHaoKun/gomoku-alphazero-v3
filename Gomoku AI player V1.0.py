"""Gargantua Gomoku desktop application / Gargantua 五子棋桌面程序。

Architecture / 代码架构
-----------------------
This is the user-facing entry point.  It owns the Pygame event loop, board
rendering, color selection, move history, undo support, and asynchronous calls
to ``alphazero_training.play_agent.AlphaZeroGomokuAgent``.  The trained model
and search stay outside the rendering loop so the window remains responsive.

本文件是用户运行的入口，负责 Pygame 事件循环、棋盘绘制、黑白选择、落子历史、
悔棋，以及异步调用 ``AlphaZeroGomokuAgent``。模型推理和搜索不在绘图主循环中
执行，因此 AI 思考时窗口仍能响应。

Key algorithms / 重要算法
-------------------------
* Exact one-move win and mandatory-block checks protect every committed move.
* The trained policy-value network and MCTS supply the normal AI decision.
* A handcrafted shape-scoring heuristic is retained only as an error fallback.
* A generation number rejects stale worker results after restart or undo.

* 一步必胜与必须封堵检查为最终落子提供战术保护。
* 正常决策来自策略-价值神经网络与蒙特卡洛树搜索（MCTS）。
* 人工棋形评分只在模型加载或推理失败时作为后备方案。
* 棋局代号会丢弃重开或悔棋后返回的过期后台结果。
"""

from pathlib import Path
import math
import os
import queue
import threading

import pygame,sys

GRID_NULL,GRID_BLACK,GRID_WHITE = 0,1,2
SPEED_X = [1,0,1,-1]
SPEED_Y = [0,1,1,1]
APP_TITLE = 'Gargantua'
APP_TAGLINE = 'GOMOKU ENGINE'
APP_CAPTION = 'Gargantua - Gomoku AI'

# ---------------------------------------------------------------------------
# Layout / 版面
# ---------------------------------------------------------------------------
# The interface is designed in the base units below and then drawn natively at
# ``UI_SCALE`` times that size: every rectangle, line, glyph and stone is
# rendered at its final pixel size, so nothing is ever bitmap-scaled.  A
# stretched bitmap is either blurry (linear) or jagged (nearest); rendering at
# the target size avoids both.
#
# 界面先用下面的基准单位排版，再按 UI_SCALE 原生绘制：矩形、线条、文字和棋子
# 都直接以最终像素尺寸生成，全程不做位图缩放。位图放大要么发虚（线性插值）
# 要么有锯齿（最近邻），原生绘制两者都不会出现。
BOARD_ORDER = 19
BASE_CELL = 38                      # grid pitch / 棋盘格距
BASE_PAD = 36                       # quiet zone inside the board card
BASE_MARGIN = 44
BASE_GAP = 32
BASE_PANEL_W = 392
BASE_BOARD_SURF = (BOARD_ORDER - 1) * BASE_CELL + BASE_PAD * 2
BASE_WIDTH = BASE_MARGIN * 2 + BASE_BOARD_SURF + BASE_GAP + BASE_PANEL_W
BASE_HEIGHT = BASE_MARGIN * 2 + BASE_BOARD_SURF


def enable_dpi_awareness():
    """Ask Windows for real pixels so the window is not bitmap-stretched.

    在 Windows 上声明 DPI 感知，避免系统把窗口放大导致模糊。
    """
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def detect_ui_scale():
    """How many physical pixels one design unit should occupy.

    A DPI-aware window is measured in physical pixels, so on a 200% display a
    1:1 frame would cover only a quarter of the screen.  The display scaling
    factor is therefore used as the drawing scale, clamped so the window still
    fits the desktop.  ``GOMOKU_UI_SCALE`` overrides it.

    DPI 感知的窗口以物理像素计量，200% 缩放下 1:1 的画面只占屏幕四分之一。
    因此这里以系统缩放比例作为绘制倍率，并限制在桌面放得下的范围内。
    可用环境变量 ``GOMOKU_UI_SCALE`` 覆盖。
    """
    override = os.environ.get('GOMOKU_UI_SCALE')
    if override:
        try:
            return max(0.5,min(float(override),4.0))
        except ValueError:
            print('Ignoring invalid GOMOKU_UI_SCALE:',override)
    if sys.platform != 'win32':
        return 1.0
    enable_dpi_awareness()
    try:
        import ctypes
        user32 = ctypes.windll.user32
        scale = user32.GetDpiForSystem() / 96.0
        screen_w,screen_h = user32.GetSystemMetrics(0),user32.GetSystemMetrics(1)
    except Exception:
        return 1.0
    # Leave room for the title bar and the taskbar.
    scale = min(scale,(screen_w - 48) / BASE_WIDTH,(screen_h - 120) / BASE_HEIGHT)
    return max(0.6,min(scale,3.0))


UI_SCALE = detect_ui_scale()


def S(value):
    """Convert a design unit into physical pixels. 把设计单位换算成物理像素。"""
    return int(round(value * UI_SCALE))


BOARD_SIZE = S(BASE_CELL)           # grid pitch / 棋盘格距
BOARD_PAD = S(BASE_PAD)
BOARD_X0,BOARD_Y0 = BOARD_PAD,BOARD_PAD
BOARD_SPAN = (BOARD_ORDER - 1) * BOARD_SIZE
BOARD_SURF = BOARD_SPAN + BOARD_PAD * 2
STONE_D = BOARD_SIZE - S(3)
MARGIN = S(BASE_MARGIN)
BOARD_X,BOARD_Y = MARGIN,MARGIN
PANEL_GAP = S(BASE_GAP)
PANEL_X = BOARD_X + BOARD_SURF + PANEL_GAP
PANEL_W = S(BASE_PANEL_W)
SCREEN_WIDTH = PANEL_X + PANEL_W + MARGIN
SCREEN_HEIGHT = BOARD_Y + BOARD_SURF + MARGIN

# Side panel rhythm, top to bottom / 右侧信息栏的纵向节奏
MARK_SIZE = S(60)
MARK_Y = MARGIN
WORD_Y = S(112)
TAG_Y = S(172)
RULE_Y = S(198)
BTN_H = S(48)
BTN_ROW1_Y = S(529)
BTN_ROW2_Y = S(589)
KEYS_Y = S(683)

# ---------------------------------------------------------------------------
# Palette / 配色
# ---------------------------------------------------------------------------
# Warm paper background, near-black ink and a single clay accent, in the spirit
# of Anthropic's brand surfaces; the board keeps a slightly deeper paper tone so
# the stones read as objects sitting on it.
INK        = (23,22,20)
INK_SOFT   = (92,88,80)
INK_MUTED  = (141,135,124)
INK_FAINT  = (186,180,167)
BG         = (240,238,230)
CARD       = (250,249,245)
BORDER     = (224,219,206)
HAIRLINE   = (232,228,217)
ACCENT     = (198,106,76)
ACCENT_SOFT= (233,199,186)
ACCENT_TINT= (247,236,231)
BOARD_BG   = (232,224,207)
BOARD_LINE = (176,164,141)
BOARD_STAR = (128,116,96)
WIN_GREEN  = (86,122,86)

# Fonts are loaded from files rather than through ``pygame.font.SysFont``:
# enumerating the Windows font registry crashes on some machines, and the file
# lookup also guarantees we get the exact weight we asked for.
# 直接按文件加载字体，既避开 SysFont 在部分 Windows 上的崩溃，也能拿到准确字重。
SANS = ('segoeui.ttf','selawk.ttf','arial.ttf','DejaVuSans.ttf')
SANS_MED = ('seguisb.ttf','segoeuisb.ttf','selawksb.ttf','segoeuib.ttf',
            'arialbd.ttf','DejaVuSans-Bold.ttf')
SERIF = ('georgia.ttf','constan.ttf','pala.ttf','times.ttf','DejaVuSerif.ttf')
FONT_DIRS = (
    Path(sys.executable).parent,
    Path('C:/Windows/Fonts'),
    Path.home() / 'AppData/Local/Microsoft/Windows/Fonts',
    Path('/usr/share/fonts/truetype/dejavu'),
    Path('/Library/Fonts'),
)
_FONT_CACHE = {}


def load_font(files,size):
    """Return a font for the first available file, or the pygame default."""
    key = (files,size)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    font = None
    for directory in FONT_DIRS:
        for name in files:
            path = directory / name
            try:
                if path.is_file():
                    font = pygame.font.Font(str(path),size)
                    break
            except OSError:
                continue
        if font is not None:
            break
    if font is None:
        font = pygame.font.Font(None,int(size * 1.32))
    _FONT_CACHE[key] = font
    return font

def is_five(grid, x, y, flag):
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal Right, Diagonal Left
    BOARD_ORDER = len(grid)  # Assuming square grid

    def check_direction(dx, dy):
        count = 1  # Start with 1 for the current piece
        # Check in the positive direction
        for i in range(1, 5):
            nx, ny = x + dx * i, y + dy * i
            if 0 <= nx < BOARD_ORDER and 0 <= ny < BOARD_ORDER and grid[ny][nx] == flag:
                count += 1
            else:
                break
        # Check in the negative direction
        for i in range(1, 5):
            nx, ny = x - dx * i, y - dy * i
            if 0 <= nx < BOARD_ORDER and 0 <= ny < BOARD_ORDER and grid[ny][nx] == flag:
                count += 1
            else:
                break
        return count >= 5

    for dx, dy in directions:
        if check_direction(dx, dy):
            return True
    return False
def board_full(grid):
    return all(cell != GRID_NULL for row in grid for cell in row)


def would_win(grid, x, y, flag):
    if grid[y][x] != GRID_NULL:
        return False
    grid[y][x] = flag
    won = is_five(grid, x, y, flag)
    grid[y][x] = GRID_NULL
    return won


def choose_forced_move(grid, ai_flag=GRID_BLACK):
    """Return an immediate AI win or the opponent's immediate winning point."""
    opponent_flag = GRID_BLACK + GRID_WHITE - ai_flag
    moves = [
        (x, y)
        for y in range(BOARD_ORDER)
        for x in range(BOARD_ORDER)
        if grid[y][x] == GRID_NULL
    ]
    for x, y in moves:
        if would_win(grid, x, y, ai_flag):
            return x, y
    for x, y in moves:
        if would_win(grid, x, y, opponent_flag):
            return x, y
    return None


def _count_side(grid, x, y, dx, dy, flag):
    count = 0
    nx, ny = x + dx, y + dy
    while 0 <= nx < BOARD_ORDER and 0 <= ny < BOARD_ORDER and grid[ny][nx] == flag:
        count += 1
        nx += dx
        ny += dy
    is_open = (
        0 <= nx < BOARD_ORDER
        and 0 <= ny < BOARD_ORDER
        and grid[ny][nx] == GRID_NULL
    )
    return count, is_open


def _shape_score(length, open_ends):
    if length >= 5:
        return 1000000
    if length == 4:
        return 120000 if open_ends == 2 else 15000 if open_ends == 1 else 0
    if length == 3:
        return 8000 if open_ends == 2 else 800 if open_ends == 1 else 0
    if length == 2:
        return 500 if open_ends == 2 else 80 if open_ends == 1 else 0
    return 10 if length == 1 and open_ends == 2 else 0


def score_move(grid, x, y, flag):
    if grid[y][x] != GRID_NULL:
        return -1

    total = 0
    useful_directions = 0
    for dx, dy in zip(SPEED_X, SPEED_Y):
        left_count, left_open = _count_side(grid, x, y, -dx, -dy, flag)
        right_count, right_open = _count_side(grid, x, y, dx, dy, flag)
        length = 1 + left_count + right_count
        open_ends = int(left_open) + int(right_open)
        total += _shape_score(length, open_ends)
        if length >= 2 and open_ends:
            useful_directions += 1

    if useful_directions >= 2:
        total += useful_directions * 180
    return total


def choose_ai_move(grid, ai_flag=GRID_BLACK):
    """Choose a legal move with immediate attack, defence, and shape scoring."""
    opponent_flag = GRID_BLACK + GRID_WHITE - ai_flag
    center = BOARD_ORDER // 2
    moves = [
        (x, y)
        for y in range(BOARD_ORDER)
        for x in range(BOARD_ORDER)
        if grid[y][x] == GRID_NULL
    ]
    moves.sort(
        key=lambda point: (
            abs(point[0] - center) + abs(point[1] - center),
            point[1],
            point[0],
        )
    )
    if not moves:
        return None

    forced = choose_forced_move(grid, ai_flag)
    if forced is not None:
        return forced

    best_move = moves[0]
    best_value = float("-inf")
    for x, y in moves:
        attack = score_move(grid, x, y, ai_flag)
        defence = score_move(grid, x, y, opponent_flag)
        center_bonus = BOARD_ORDER - abs(x - center) - abs(y - center)
        value = attack * 1.05 + defence * 1.15 + center_bonus
        if value > best_value:
            best_value = value
            best_move = (x, y)
    return best_move

def RT_draw_txt(scr,fnt,cls,txt,x,y):
    pic = fnt.render(txt,True,cls)
    scr.blit(pic,(x,y))
    return


# ---------------------------------------------------------------------------
# Drawing helpers / 绘图工具
# ---------------------------------------------------------------------------
def _lerp(c0,c1,t):
    return tuple(int(a + (b - a) * t) for a,b in zip(c0,c1))


def draw_text(scr,fnt,color,text,x,y,align='left'):
    """Blit ``text`` and return its rect. ``align`` is left/center/right."""
    pic = fnt.render(text,True,color)
    rect = pic.get_rect()
    if align == 'center':
        rect.midtop = (x,y)
    elif align == 'right':
        rect.topright = (x,y)
    else:
        rect.topleft = (x,y)
    scr.blit(pic,rect)
    return rect


def draw_tracked(scr,fnt,color,text,x,y,tracking=2,align='left'):
    """Draw letter-spaced small caps, used for the quiet section labels."""
    glyphs = [(fnt.render(ch,True,color)) for ch in text]
    width = sum(g.get_width() for g in glyphs) + tracking * max(0,len(glyphs) - 1)
    if align == 'center':
        x -= width // 2
    elif align == 'right':
        x -= width
    for glyph in glyphs:
        scr.blit(glyph,(x,y))
        x += glyph.get_width() + tracking
    return width


def aa_circle(surf,color,center,radius,width=0):
    """Antialiased circle, falling back to the plain renderer if needed."""
    x,y,r = int(center[0]),int(center[1]),int(radius)
    try:
        from pygame import gfxdraw
    except ImportError:
        pygame.draw.circle(surf,color,(x,y),r,width)
        return
    if width <= 0:
        gfxdraw.filled_circle(surf,x,y,r,color)
        gfxdraw.aacircle(surf,x,y,r,color)
    else:
        for i in range(width):
            gfxdraw.aacircle(surf,x,y,r - i,color)


def fit_text(fnt,text,max_width):
    """Trim ``text`` with an ellipsis so it never overflows its card."""
    if fnt.size(text)[0] <= max_width:
        return text
    while text and fnt.size(text + '...')[0] > max_width:
        text = text[:-1]
    return text + '...'


def make_shadow(size,radius,spread=10,alpha=6,offset=4):
    """A soft drop shadow built from stacked translucent rounded rectangles."""
    w,h = size
    surf = pygame.Surface((w + spread * 2,h + spread * 2 + offset),pygame.SRCALPHA)
    for i in range(spread,0,-1):
        pygame.draw.rect(
            surf,(64,56,40,alpha),
            (spread - i,spread - i + offset,w + i * 2,h + i * 2),
            border_radius=radius + i,
        )
    return surf


def draw_card(scr,rect,fill=CARD,border=BORDER,radius=16,width=1):
    pygame.draw.rect(scr,fill,rect,border_radius=radius)
    if border is not None:
        pygame.draw.rect(scr,border,rect,width,border_radius=radius)
    return rect


def make_stone(diameter,edge,core,spec=110,rim=None):
    """Render one stone with a soft radial gradient, rim light and shadow.

    Drawn at 4x and scaled down, which gives clean antialiased edges without
    depending on any image asset.  以 4 倍尺寸绘制后缩小，得到平滑抗锯齿边缘。
    """
    ss = 4
    d = diameter * ss
    pad = int(d * 0.20)
    size = d + pad * 2
    surf = pygame.Surface((size,size),pygame.SRCALPHA)
    cx = cy = size // 2
    r = d // 2

    contact = max(1,int(pad * 0.85))
    for i in range(contact,0,-1):
        alpha = int(30 * (1 - i / contact) ** 1.4) + 3
        pygame.draw.circle(surf,(46,38,26,alpha),(cx,cy + int(d * 0.05)),r + i)

    for i in range(r,0,-1):
        t = i / r
        offset = int(-(1 - t) * d * 0.11)
        pygame.draw.circle(surf,_lerp(core,edge,t ** 1.5),(cx + offset,cy + offset),i)

    if rim is not None:
        pygame.draw.circle(surf,rim,(cx,cy),r,max(1,ss // 2))

    if spec:
        gloss = pygame.Surface((size,size),pygame.SRCALPHA)
        hr = max(2,int(r * 0.36))
        for i in range(hr,0,-1):
            alpha = int(spec * (1 - i / hr) ** 1.5)
            pygame.draw.circle(
                gloss,(255,255,255,alpha),
                (cx - int(r * 0.34),cy - int(r * 0.38)),i,
            )
        surf.blit(gloss,(0,0))
    return pygame.transform.smoothscale(surf,(size // ss,size // ss))


def make_black_hole(size):
    """The Gargantua mark: an event horizon ringed by a lensed accretion disk.

    Gargantua 是黑洞的名字，用视界与被引力透镜弯折的吸积盘作为标志。
    """
    ss = 3
    s = size * ss
    surf = pygame.Surface((s,s),pygame.SRCALPHA)
    c = (s - 1) / 2.0
    core_r = s * 0.245
    ring_r = s * 0.281                  # brightest radius of the photon ring
    sigma = s * 0.048                   # how soft the ring falls off
    reach = s * 0.50

    pygame.draw.circle(surf,(11,10,11,255),(round(c),round(c)),int(core_r) + 1)

    # The ring is shaded per pixel: a soft radial band, brightened on the side
    # rotating toward the viewer the way Doppler beaming lights one edge.
    # 逐像素着色：径向柔和光带，并让朝向观察者的一侧更亮（多普勒增亮）。
    warm,bright = (214,124,84),(255,222,182)
    for py in range(s):
        dy = py - c
        for px in range(s):
            dx = px - c
            r = math.hypot(dx,dy)
            if r < core_r or r > reach:
                continue
            band = math.exp(-((r - ring_r) / sigma) ** 2)
            if band < 0.012:
                continue
            beam = 0.5 + 0.5 * (dy / r)
            color = _lerp(warm,bright,beam ** 1.4)
            surf.set_at((px,py),(*color,min(255,int(255 * band * (0.5 + 0.5 * beam)))))
    return pygame.transform.smoothscale(surf,(size,size))

def RT_get_flag_beads(grid,x,y,man,flag):
    beadsNum,powerNum = 1,0
    for i in range(-1,-5,-1):
        tx,ty = x + i * SPEED_X[flag],y + i * SPEED_Y[flag]
        if tx < 0 or tx >= BOARD_ORDER or ty < 0 or ty >= BOARD_ORDER:
            break
        if grid[ty][tx]!= man:
            powerNum += (grid[ty][tx] == GRID_NULL)
            break
        beadsNum = beadsNum + 1
    for i in range(1,5,1):
        tx,ty = x + i * SPEED_X[flag],y + i * SPEED_Y[flag]
        if tx < 0 or tx >= BOARD_ORDER or ty < 0 or ty >= BOARD_ORDER:
            break
        if grid[ty][tx] != man:
            powerNum += (grid[ty][tx] == GRID_NULL)
            break
        beadsNum = beadsNum + 1
    if beadsNum >= 5:
        beadsNum = 5
    return [beadsNum,powerNum]


ASSESS_WIN,ASSESS_ANS,ASSESS_COUNT = 2,1,0
def RT_get_assess_value(countList):
    assess,value = 0,0
    if ([5,2] in countList) or ([5,1] in countList) or ([5,0] in countList):
        assess,value = ASSESS_WIN,200
    elif [4,2] in countList:
        assess,value = ASSESS_ANS,100
    else:
        value += countList.count([4,1]) * 70
        value += countList.count([3,2]) * 60
        value += countList.count([3,1]) * 30
        value += countList.count([2,2]) * 20
        value += countList.count([2,1]) * 15
        assess = ASSESS_COUNT
    return assess,value
class CLS_assess(object):
    def __init__(self,x,y):
        self.x,self.y = x,y
        self.bAssess,self.wAssess = 0,0
        self.bValue,self.wValue = 0,0
        self.bCount = [[0,0],[0,0],[0,0],[0,0]]
        self.wCount = [[0,0],[0,0],[0,0],[0,0]]
        return
    def beads(self,grid):
        for flag in (0,1,2,3):
            self.bCount[flag]\
            = RT_get_flag_beads(grid,self.x,self.y,GRID_BLACK,flag)
            self.wCount[flag]\
            = RT_get_flag_beads(grid,self.x,self.y,GRID_WHITE,flag)
        return
    def assess(self,grid):
        self.beads(grid)
        self.bAssess,self.bValue = RT_get_assess_value(self.bCount)
        self.wAssess,self.wValue = RT_get_assess_value(self.wCount)
        return

class CLS_gomoku(object):
    def draw_board(self):
        """Pre-render the static board: paper, grid, star points, coordinates."""
        self.board = pygame.Surface((BOARD_SURF,BOARD_SURF),pygame.SRCALPHA)
        pygame.draw.rect(self.board,BOARD_BG,self.board.get_rect(),border_radius=S(18))
        o,L = BOARD_X0,BOARD_X0 + BOARD_SPAN
        hair = max(1,S(1))
        for i in range(BOARD_ORDER):
            p = o + i * BOARD_SIZE
            pygame.draw.line(self.board,BOARD_LINE,(o,p),(L,p),hair)
            pygame.draw.line(self.board,BOARD_LINE,(p,o),(p,L),hair)
        pygame.draw.rect(self.board,_lerp(BOARD_LINE,INK,0.30),
                         (o,o,BOARD_SPAN,BOARD_SPAN),max(2,S(2)))
        for star_y in (3,9,15):
            for star_x in (3,9,15):
                aa_circle(self.board,BOARD_STAR,
                          (o + star_x * BOARD_SIZE,o + star_y * BOARD_SIZE),S(4))
        label_color = _lerp(BOARD_LINE,INK,0.35)
        half = self.fontCoord.get_height() // 2
        for i in range(BOARD_ORDER):
            p = o + i * BOARD_SIZE
            letter,number = chr(ord('A') + i),str(i + 1)
            draw_text(self.board,self.fontCoord,label_color,letter,p,o - S(25),'center')
            draw_text(self.board,self.fontCoord,label_color,letter,p,L + S(9),'center')
            draw_text(self.board,self.fontCoord,label_color,number,o - S(13),p - half,'right')
            draw_text(self.board,self.fontCoord,label_color,number,L + S(13),p - half)
        return

    def _build_backdrop(self):
        """Render every static pixel once: paper, shadows, cards, wordmark."""
        self.backdrop = pygame.Surface((SCREEN_WIDTH,SCREEN_HEIGHT))
        scr = self.backdrop
        scr.fill(BG)

        hair = max(1,S(1))
        board_rect = pygame.Rect(self.x0,self.y0,BOARD_SURF,BOARD_SURF)
        shadow = make_shadow(board_rect.size,S(18),spread=S(14),alpha=5,offset=S(6))
        scr.blit(shadow,(board_rect.x - S(14),board_rect.y - S(14)))
        scr.blit(self.board,board_rect.topleft)

        scr.blit(self.mark,(PANEL_X,MARK_Y))
        draw_text(scr,self.fontWord,INK,APP_TITLE,PANEL_X - S(2),WORD_Y)
        draw_tracked(scr,self.fontLabel,ACCENT,APP_TAGLINE,PANEL_X,TAG_Y,tracking=S(3))
        pygame.draw.line(scr,BORDER,(PANEL_X,RULE_Y),(PANEL_X + PANEL_W,RULE_Y),hair)

        for rect in (self.status_card,self.model_card,self.stats_card):
            scr.blit(make_shadow(rect.size,S(16),spread=S(10),offset=S(4)),
                     (rect.x - S(10),rect.y - S(10)))
            draw_card(scr,rect)

        draw_tracked(scr,self.fontLabel,INK_MUTED,'STATUS',
                     self.status_card.x + S(20),self.status_card.y + S(18),tracking=S(2))
        draw_tracked(scr,self.fontLabel,INK_MUTED,'ENGINE',
                     self.model_card.x + S(20),self.model_card.y + S(18),tracking=S(2))
        pygame.draw.line(scr,HAIRLINE,
                         (self.stats_card.centerx,self.stats_card.y + S(18)),
                         (self.stats_card.centerx,self.stats_card.bottom - S(18)),hair)

        pygame.draw.line(scr,BORDER,(PANEL_X,KEYS_Y - S(22)),
                         (PANEL_X + PANEL_W,KEYS_Y - S(22)),hair)
        draw_tracked(scr,self.fontLabel,INK_MUTED,'SHORTCUTS',PANEL_X,KEYS_Y,tracking=S(2))
        keys = (
            ('B / W','choose your color'),
            ('U / Bksp','undo your last turn'),
            ('R / Enter','restart the game'),
            ('C','back to color choice'),
            ('S','toggle shape scores'),
        )
        for row,(key,meaning) in enumerate(keys):
            y = KEYS_Y + S(22) + row * S(20)
            draw_text(scr,self.fontKey,INK_SOFT,key,PANEL_X,y)
            draw_text(scr,self.fontHint,INK_MUTED,meaning,PANEL_X + S(92),y + S(1))
        return

    def __init__(self,bPic,wPic,x0=BOARD_X,y0=BOARD_Y,ai_agent=None,game_logger=None):
        self.bMan,self.wMan = bPic,wPic
        # Small stones for the identity pill; downscaling stays smooth.
        chip_d = S(22)
        self.bChip = pygame.transform.smoothscale(bPic,(chip_d,chip_d))
        self.wChip = pygame.transform.smoothscale(wPic,(chip_d,chip_d))
        self.fontCoord = load_font(SANS,S(12))
        self.fontLabel = load_font(SANS_MED,S(11))
        self.fontScore = load_font(SANS,S(11))
        self.fontMove = load_font(SANS,S(13))
        self.fontHint = load_font(SANS,S(13))
        self.fontKey = load_font(SANS_MED,S(13))
        self.fontBody = load_font(SANS,S(14))
        self.fontValue = load_font(SANS_MED,S(17))
        self.fontStatus = load_font(SANS_MED,S(22))
        self.fontButton = load_font(SANS_MED,S(15))
        self.fontWord = load_font(SERIF,S(44))
        self.fontWin = load_font(SERIF,S(42))
        self.font = self.fontBody
        self.x0,self.y0 = x0,y0
        self.mark = make_black_hole(MARK_SIZE)
        self.status_card = pygame.Rect(PANEL_X,S(214),PANEL_W,S(88))
        self.model_card = pygame.Rect(PANEL_X,S(314),PANEL_W,S(104))
        self.stats_card = pygame.Rect(PANEL_X,S(430),PANEL_W,S(78))
        self.black_button = pygame.Rect(PANEL_X,BTN_ROW1_Y,PANEL_W,BTN_H)
        self.white_button = pygame.Rect(PANEL_X,BTN_ROW2_Y,PANEL_W,BTN_H)
        self.undo_button = pygame.Rect(PANEL_X,BTN_ROW2_Y,PANEL_W // 2 - S(7),BTN_H)
        self.restart_button = pygame.Rect(
            PANEL_X + PANEL_W // 2 + S(7),BTN_ROW2_Y,PANEL_W // 2 - S(7),BTN_H
        )
        self.mouse = (-1,-1)
        self.hover_cell = None
        self.draw_board()
        self._build_backdrop()
        self.ai_agent = ai_agent
        self.ai_results = queue.Queue()
        self.ai_thinking = False
        self.game_generation = 0
        self.game_logger = game_logger
        self._game_logged = False
        self.ai_label = ai_agent.label if ai_agent is not None else 'Heuristic fallback'
        self.ai_model_label = (
            getattr(ai_agent,'model_label',self.ai_label)
            if ai_agent is not None else self.ai_label
        )
        self.ai_search_label = (
            getattr(ai_agent,'search_label','') if ai_agent is not None else ''
        )
        self.sysStat = 0
        self.assessFlag = False
        self.winner = -1
        self.human_color = None
        self.ai_color = None
        self.turn = GRID_BLACK
        self.grid_init(None)
    def grid_init(self,human_color=None):
        self.grid = []
        self.move_numbers = []
        for y in range(BOARD_ORDER):
            line = [GRID_NULL] * BOARD_ORDER
            self.grid.append(line)
            self.move_numbers.append([0] * BOARD_ORDER)
        self.move_count = 0
        self.move_history = []
        self.assessList = []
        for y in range(BOARD_ORDER):
            line = []
            for x in range(BOARD_ORDER):
                score = CLS_assess(x,y)
                line.append(score)
            self.assessList.append(line)
        self.bMaxAssess,self.bMaxValue,self.bpX,self.bpY = 0,0,9,9
        self.wMaxAssess,self.wMaxValue,self.wpX,self.wpY = 0,0,-1,-1
        self.SumValue,self.pX,self.pY = 0,-1,-1
        self.human_color = human_color
        self.ai_color = (
            GRID_BLACK + GRID_WHITE - human_color
            if human_color in (GRID_BLACK,GRID_WHITE)
            else None
        )
        self.turn = GRID_BLACK
        self.last_move = None
        self.last_ai_move = None
        self.game_generation += 1
        self.ai_thinking = False
        self.winner = -1
        self.sysStat = 0
        self._game_logged = False
        return
    def select_color(self,human_color):
        if human_color not in (GRID_BLACK,GRID_WHITE):
            return
        self._archive_unfinished_game('color_changed')
        self.grid_init(human_color)
        if self.ai_color == GRID_BLACK:
            self._start_ai_move()
    def can_undo(self):
        return (
            self.human_color in (GRID_BLACK,GRID_WHITE)
            and any(color == self.human_color for _,_,color in self.move_history)
        )
    def undo(self):
        """Undo the human's latest turn and the AI reply that followed it."""
        if not self.can_undo():
            return False

        # Invalidate any move still being calculated on the old position.
        self.game_generation += 1
        self.ai_thinking = False

        human_move_index = max(
            index
            for index,(_,_,color) in enumerate(self.move_history)
            if color == self.human_color
        )
        for x,y,_ in self.move_history[human_move_index:]:
            self.grid[y][x] = GRID_NULL
        del self.move_history[human_move_index:]

        self.move_numbers = [[0] * BOARD_ORDER for _ in range(BOARD_ORDER)]
        for number,(x,y,_) in enumerate(self.move_history,start=1):
            self.move_numbers[y][x] = number
        self.move_count = len(self.move_history)
        self.last_move = (
            self.move_history[-1][:2] if self.move_history else None
        )
        self.last_ai_move = next(
            (
                (x,y)
                for x,y,color in reversed(self.move_history)
                if color == self.ai_color
            ),
            None,
        )
        self.winner = -1
        self.sysStat = 0
        self._game_logged = False
        self.turn = self.human_color
        self.grid_assess()
        return True

    def _archive_game(self,winner,termination):
        if (
            self._game_logged
            or self.game_logger is None
            or not self.move_history
            or self.ai_color not in (GRID_BLACK,GRID_WHITE)
        ):
            return None
        # Mark first so a disk error cannot cause duplicate archives later.
        self._game_logged = True
        try:
            saved = self.game_logger.record_game(
                self.move_history,
                winner=winner,
                ai_color=self.ai_color,
                model_label=self.ai_model_label,
                search_label=self.ai_search_label,
                termination=termination,
            )
            print('Game replay saved:',saved.replay_path)
            if saved.pending_replay_path is not None:
                print('AI loss added to pending training library:',saved.pending_replay_path)
            return saved
        except Exception as exc:
            print('Could not save game replay:',exc)
            return None

    def _archive_unfinished_game(self,termination):
        if self.sysStat != 1:
            return self._archive_game(None,termination)
        return None

    def _finish_game(self,winner,message=None):
        self.winner = winner
        self.sysStat = 1
        if message:
            print(message)
        return self._archive_game(winner,'completed')

    def close(self):
        """Archive an in-progress game before the application exits."""
        self._archive_unfinished_game('window_closed')
    def cell_center(self,x,y):
        """Screen center of intersection ``(x,y)``. 交点的屏幕坐标。"""
        return (self.x0 + BOARD_X0 + x * BOARD_SIZE,
                self.y0 + BOARD_Y0 + y * BOARD_SIZE)

    def set_pointer(self,mx,my):
        """Track the cursor so the board can preview the move under it."""
        self.mouse = (mx,my)
        self.hover_cell = None
        if (self.human_color is None or self.sysStat == 1
                or self.ai_thinking or self.turn != self.human_color):
            return
        cell = self.point_to_cell(mx,my)
        if cell is not None and self.grid[cell[1]][cell[0]] == GRID_NULL:
            self.hover_cell = cell

    def point_to_cell(self,mx,my):
        """Map a click to an intersection, ignoring clicks off the board."""
        x = round((mx - self.x0 - BOARD_X0) / BOARD_SIZE)
        y = round((my - self.y0 - BOARD_Y0) / BOARD_SIZE)
        if not (0 <= x < BOARD_ORDER and 0 <= y < BOARD_ORDER):
            return None
        cx,cy = self.cell_center(x,y)
        if math.hypot(mx - cx,my - cy) > BOARD_SIZE * 0.62:
            return None
        return x,y

    def draw_chess(self,scr):
        radius = STONE_D // 2
        for y in range(BOARD_ORDER):
            for x in range(BOARD_ORDER):
                piece_color = self.grid[y][x]
                cx,cy = self.cell_center(x,y)
                if piece_color == GRID_NULL:
                    if self.assessFlag:
                        pnt = self.assessList[y][x]
                        if pnt.bAssess > 0 or pnt.bValue > 0:
                            RT_draw_txt(scr,self.fontScore,(48,45,41),
                                        str(pnt.bAssess) + ',' + str(pnt.bValue),
                                        cx - S(17),cy - S(13))
                        if pnt.wAssess > 0 or pnt.wValue > 0:
                            RT_draw_txt(scr,self.fontScore,(250,249,245),
                                        str(pnt.wAssess) + ',' + str(pnt.wValue),
                                        cx - S(17),cy + S(1))
                    continue
                stone = self.bMan if piece_color == GRID_BLACK else self.wMan
                scr.blit(stone,stone.get_rect(center=(cx,cy)))
                if (x,y) == self.last_ai_move:
                    aa_circle(scr,ACCENT,(cx,cy),radius + S(4),max(2,S(2)))
                elif (x,y) == self.last_move:
                    aa_circle(scr,INK_MUTED,(cx,cy),radius + S(4),max(1,S(1)))
                move_number = self.move_numbers[y][x]
                if move_number:
                    number_color = (233,229,220) if piece_color == GRID_BLACK else (86,81,73)
                    number_pic = self.fontMove.render(str(move_number),True,number_color)
                    scr.blit(number_pic,number_pic.get_rect(center=(cx,cy)))
        if self.hover_cell is not None:
            hx,hy = self.hover_cell
            ghost = pygame.Surface((STONE_D + S(2),STONE_D + S(2)),pygame.SRCALPHA)
            center = (ghost.get_width() // 2,ghost.get_height() // 2)
            tint = ((26,24,22,48) if self.human_color == GRID_BLACK
                    else (255,253,248,140))
            aa_circle(ghost,tint,center,radius)
            aa_circle(ghost,(*INK_MUTED,150),center,radius,max(1,S(1)))
            scr.blit(ghost,ghost.get_rect(center=self.cell_center(hx,hy)))
        return

    def _status_lines(self):
        """Headline, accent color and supporting line for the status card."""
        you = 'black' if self.human_color == GRID_BLACK else 'white'
        if self.human_color is None:
            return 'Choose a color',ACCENT,'Black opens the game'
        if self.sysStat == 1:
            if self.winner == GRID_NULL:
                return 'Draw',INK_SOFT,'The board is full'
            if self.winner == self.human_color:
                return 'You win',WIN_GREEN,'Five in a row on move %d' % self.move_count
            return 'Gargantua wins',ACCENT,'Five in a row on move %d' % self.move_count
        if self.ai_thinking:
            return 'Gargantua is thinking',ACCENT,'Searching the tree'
        return 'Your move',INK,'Move %d, you play %s' % (self.move_count + 1,you)

    def _draw_status(self,scr):
        card = self.status_card
        headline,tone,detail = self._status_lines()
        text_y = card.y + S(34)
        aa_circle(scr,tone,(card.x + S(26),text_y + self.fontStatus.get_height() // 2),S(5))
        draw_text(scr,self.fontStatus,INK,
                  fit_text(self.fontStatus,headline,card.w - S(62)),card.x + S(42),text_y)
        draw_text(scr,self.fontHint,INK_MUTED,detail,card.x + S(42),card.y + S(62))

    def _draw_model(self,scr):
        card = self.model_card
        inner = card.w - S(40)
        draw_text(scr,self.fontValue,INK,
                  fit_text(self.fontValue,self.ai_model_label,inner),
                  card.x + S(20),card.y + S(38))
        second = self.ai_search_label or self.ai_label
        draw_text(scr,self.fontHint,INK_SOFT,
                  fit_text(self.fontHint,second,inner),card.x + S(20),card.y + S(64))
        opponent = ('plays white' if self.ai_color == GRID_WHITE
                    else 'plays black' if self.ai_color == GRID_BLACK else 'waiting')
        draw_text(scr,self.fontHint,INK_FAINT,opponent,card.x + S(20),card.y + S(82))

    def _draw_stats(self,scr):
        card = self.stats_card
        last = '—'
        if self.last_move is not None:
            last = chr(ord('A') + self.last_move[0]) + str(self.last_move[1] + 1)
        columns = (
            (card.x + S(20),'MOVES',str(self.move_count)),
            (card.centerx + S(20),'LAST MOVE',last),
        )
        for x,label,value in columns:
            draw_tracked(scr,self.fontLabel,INK_MUTED,label,x,card.y + S(22),tracking=S(2))
            draw_text(scr,self.fontValue,INK,value,x,card.y + S(42))

    def _draw_button(self,scr,rect,text,fill,border,label_color,radius=None):
        radius = S(14) if radius is None else radius
        if rect.collidepoint(self.mouse):
            fill = _lerp(fill,ACCENT,0.10) if fill != CARD else (255,254,251)
        pygame.draw.rect(scr,fill,rect,border_radius=radius)
        if border is not None:
            pygame.draw.rect(scr,border,rect,max(1,S(1)),border_radius=radius)
        pic = self.fontButton.render(text,True,label_color)
        scr.blit(pic,pic.get_rect(center=rect.center))

    def _draw_controls(self,scr):
        if self.human_color is None:
            self._draw_button(scr,self.black_button,'Play as black  ·  move first',
                              INK,None,(250,249,245))
            self._draw_button(scr,self.white_button,'Play as white  ·  move second',
                              CARD,BORDER,INK)
            return

        pill = pygame.Rect(PANEL_X,BTN_ROW1_Y,PANEL_W,BTN_H)
        pygame.draw.rect(scr,ACCENT_TINT,pill,border_radius=S(14))
        chip = self.bChip if self.human_color == GRID_BLACK else self.wChip
        scr.blit(chip,chip.get_rect(center=(pill.x + S(26),pill.centery)))
        you = 'black' if self.human_color == GRID_BLACK else 'white'
        draw_text(scr,self.fontBody,INK,'You play ' + you,pill.x + S(46),
                  pill.centery - S(9))
        draw_text(scr,self.fontHint,INK_MUTED,'C to switch',pill.right - S(18),
                  pill.centery - S(8),'right')

        can_undo = self.can_undo()
        self._draw_button(
            scr,self.undo_button,'Undo',
            CARD if can_undo else (245,243,236),
            BORDER if can_undo else HAIRLINE,
            INK if can_undo else INK_FAINT,
        )
        self._draw_button(scr,self.restart_button,'New game',ACCENT,None,(255,252,250))

    def _draw_result(self,scr):
        board_rect = pygame.Rect(self.x0,self.y0,BOARD_SURF,BOARD_SURF)
        veil = pygame.Surface(board_rect.size,pygame.SRCALPHA)
        pygame.draw.rect(veil,(*BG,206),veil.get_rect(),border_radius=S(18))
        scr.blit(veil,board_rect.topleft)

        card = pygame.Rect(0,0,S(420),S(160))
        card.center = board_rect.center
        scr.blit(make_shadow(card.size,S(20),spread=S(18),alpha=6,offset=S(8)),
                 (card.x - S(18),card.y - S(18)))
        draw_card(scr,card,radius=S(20))

        if self.winner == GRID_NULL:
            title,tone = 'Draw',INK_SOFT
        elif self.winner == self.human_color:
            title,tone = 'You win',WIN_GREEN
        else:
            title,tone = 'Gargantua wins',ACCENT
        draw_text(scr,self.fontWin,tone,title,card.centerx,card.y + S(34),'center')
        pygame.draw.line(scr,ACCENT_SOFT,(card.centerx - S(26),card.y + S(98)),
                         (card.centerx + S(26),card.y + S(98)),max(2,S(2)))
        draw_text(scr,self.fontHint,INK_MUTED,'Press R or New game to play again',
                  card.centerx,card.y + S(112),'center')

    def draw(self,scr):
        scr.blit(self.backdrop,(0,0))
        self.draw_chess(scr)
        self._draw_status(scr)
        self._draw_model(scr)
        self._draw_stats(scr)
        self._draw_controls(scr)
        if self.sysStat == 1:
            self._draw_result(scr)
        return
    def grid_assess(self):
        self.bMaxAssess,self.bMaxValue,self.bpX,self.bpY = 0,0,-1,-1
        self.wMaxAssess,self.wMaxValue,self.wpX,self.wpY = 0,0,-1,-1
        self.SumValue,self.pX,self.pY = 0,-1,-1
        for y in range(BOARD_ORDER):
            for x in range(BOARD_ORDER):
                if self.grid[y][x] != GRID_NULL:
                    continue
                self.assessList[y][x].assess(self.grid)
                pnt = self.assessList[y][x]
                if (pnt.bAssess > self.bMaxAssess)\
                or((pnt.bAssess == self.bMaxAssess)\
                   and (pnt.bValue > self.bMaxValue)):
                    self.bMaxAssess,self.bMaxValue,self.bpX,self.bpY = pnt.bAssess,pnt.bValue,x,y
                if (pnt.wAssess > self.wMaxAssess)\
                or((pnt.wAssess == self.wMaxAssess)\
                   and (pnt.wValue > self.wMaxValue)):
                    self.wMaxAssess,self.wMaxValue,self.wpX,self.wpY = pnt.wAssess,pnt.wValue,x,y
                if pnt.wValue + pnt.bValue > self.SumValue:
                    self.SumValue,self.pX,self.pY = pnt.wValue + pnt.bValue,x,y
        return
    def grid_policy(self):
        if self.bMaxAssess == ASSESS_WIN:
            print('ASSESS_WIN:',self.bpX,self.bpY)
            return self.bpX,self.bpY
        elif self.wMaxAssess == ASSESS_WIN:
            print('ASSESS_WIN:',self.wpX,self.wpY)
            return self.wpX,self.wpY
        elif self.bMaxAssess > ASSESS_COUNT:
            print('B ASSESS_COUNT:',self.bpX,self.bpY)
            return self.bpX,self.bpY
        elif self.wMaxAssess > ASSESS_COUNT:
            print('W ASSESS_COUNT:',self.wpX,self.wpY)
            return self.wpX,self.wpY
        else:
            print('SUM ASSESS_COUNT:',self.pX,self.pY)
            return self.pX,self.pY

    def _calculate_ai_move(self,grid_snapshot,last_move,generation,ai_color):
        error = None
        try:
            forced = choose_forced_move(grid_snapshot,ai_color)
            if forced is not None:
                move = forced
            elif self.ai_agent is None:
                move = choose_ai_move(grid_snapshot,ai_color)
            else:
                move = self.ai_agent.choose_move(grid_snapshot,last_move,ai_color=ai_color)
        except Exception as exc:
            error = str(exc)
            move = choose_ai_move(grid_snapshot,ai_color)
        self.ai_results.put((generation,move,error))

    def _start_ai_move(self):
        if self.ai_color is None or self.turn != self.ai_color or self.sysStat == 1:
            return
        self.ai_thinking = True
        generation = self.game_generation
        grid_snapshot = [row[:] for row in self.grid]
        worker = threading.Thread(
            target=self._calculate_ai_move,
            args=(grid_snapshot,self.last_move,generation,self.ai_color),
            daemon=True,
        )
        worker.start()

    def update(self):
        try:
            generation,move,error = self.ai_results.get_nowait()
        except queue.Empty:
            return
        if generation != self.game_generation:
            return
        self.ai_thinking = False
        if error:
            print('AlphaZero error; heuristic fallback used:',error)
        if self.sysStat == 1 or move is None or self.turn != self.ai_color:
            return
        # Re-check the live board immediately before committing the worker's
        # result. This prevents a stale/weak model decision from ever ignoring
        # a one-ply win or mandatory block.
        forced = choose_forced_move(self.grid,self.ai_color)
        if forced is not None and move != forced:
            print('Tactical guard override:',move,'->',forced)
            move = forced
        ai_x,ai_y = move
        if not (0 <= ai_x < BOARD_ORDER and 0 <= ai_y < BOARD_ORDER)\
        or self.grid[ai_y][ai_x] != GRID_NULL:
            fallback = choose_ai_move(self.grid,self.ai_color)
            if fallback is None:
                self._finish_game(GRID_NULL)
                return
            ai_x,ai_y = fallback
        self.grid[ai_y][ai_x] = self.ai_color
        self.move_count += 1
        self.move_numbers[ai_y][ai_x] = self.move_count
        self.move_history.append((ai_x,ai_y,self.ai_color))
        self.last_move = (ai_x,ai_y)
        self.last_ai_move = (ai_x,ai_y)
        if is_five(self.grid,ai_x,ai_y,self.ai_color):
            self._finish_game(self.ai_color,'You lose!!!')
            return
        if board_full(self.grid):
            self._finish_game(GRID_NULL)
            return
        self.turn = self.human_color
        self.grid_assess()

    def restart(self):
        self._archive_unfinished_game('restarted')
        self.grid_init(self.human_color)
        if self.ai_color == GRID_BLACK:
            self._start_ai_move()

    def mouse_down(self,mx,my):
        if self.human_color is None:
            if self.black_button.collidepoint(mx,my):
                self.select_color(GRID_BLACK)
            elif self.white_button.collidepoint(mx,my):
                self.select_color(GRID_WHITE)
            return
        if self.undo_button.collidepoint(mx,my):
            self.undo()
            return
        if self.restart_button.collidepoint(mx,my):
            self.restart()
            return
        if self.sysStat == 1 or self.ai_thinking or self.turn != self.human_color:
            return
        cell = self.point_to_cell(mx,my)
        if cell is None:
            return
        x,y = cell
        if self.grid[y][x] != GRID_NULL:
            return
        self.hover_cell = None

        self.grid[y][x] = self.human_color
        self.move_count += 1
        self.move_numbers[y][x] = self.move_count
        self.move_history.append((x,y,self.human_color))
        self.last_move = (x,y)
        if is_five(self.grid, x, y, self.human_color):
            self._finish_game(self.human_color,'You win!!!')
            return
        if board_full(self.grid):
            self._finish_game(GRID_NULL)
            return

        self.turn = self.ai_color
        self._start_ai_move()
        return
    def eventkey(self,key):
        if key == pygame.K_b:
            self.select_color(GRID_BLACK)
            return
        if key == pygame.K_w:
            self.select_color(GRID_WHITE)
            return
        if key == pygame.K_c:
            self._archive_unfinished_game('color_selection_requested')
            self.grid_init(None)
            return
        if key in (pygame.K_u,pygame.K_BACKSPACE):
            self.undo()
            return
        if key in (pygame.K_RETURN,pygame.K_r):
            self.restart()
        if key == pygame.K_s:
            self.assessFlag = not self.assessFlag
        return


#-------main--------
def main():
    enable_dpi_awareness()
    pygame.init()
    pygame.display.set_caption(APP_CAPTION)
    # SCREEN_WIDTH/HEIGHT already include UI_SCALE, so the window is created at
    # its true pixel size and every frame is drawn straight into it: no
    # transform, therefore neither blur nor stair-stepping.
    # 尺寸常量已含 UI_SCALE，窗口按真实像素创建、逐帧直接绘制，不做任何变换，
    # 因此既不会发虚也不会有锯齿。
    window = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    print('UI scale %.2f, window %dx%d' % (UI_SCALE,SCREEN_WIDTH,SCREEN_HEIGHT))
    asset_dir = Path(__file__).resolve().parent
    # Stones are generated rather than loaded, so they stay crisp at any size.
    bPic = make_stone(STONE_D,(8,8,9),(74,72,76),spec=86,rim=(96,92,90))
    wPic = make_stone(STONE_D,(219,213,200),(255,253,248),spec=150,rim=(206,199,184))
    ai_agent = None
    from alphazero_training.game_logger import GameReplayLogger
    game_logger = GameReplayLogger(asset_dir / 'alphazero_training' / 'play_logs')
    checkpoint_path = asset_dir / 'alphazero_training' / 'latest.pt'
    try:
        from alphazero_training.play_agent import AlphaZeroGomokuAgent
        # Strong mode defaults to 256 MCTS searches. Set the environment
        # variable GOMOKU_MCTS_SIMULATIONS to another positive integer for a
        # temporary speed/strength trade-off without editing this file.
        ai_agent = AlphaZeroGomokuAgent(checkpoint_path)
        print('Loaded:',ai_agent.label,'checkpoint iteration',ai_agent.iteration)
    except Exception as exc:
        print('Could not load AlphaZero model; using heuristic AI:',exc)
    gomoku = CLS_gomoku(bPic,wPic,BOARD_X,BOARD_Y,ai_agent,game_logger)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                gomoku.mouse_down(*event.pos)
            elif event.type == pygame.MOUSEMOTION:
                gomoku.set_pointer(*event.pos)
            elif event.type == pygame.KEYDOWN:
                gomoku.eventkey(event.key)
            elif event.type == pygame.QUIT:
                running = False
        gomoku.update()
        gomoku.draw(window)
        pygame.display.update()
        clock.tick(60)

    gomoku.close()
    pygame.quit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
