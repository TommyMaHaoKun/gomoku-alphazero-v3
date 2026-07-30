from pathlib import Path
import queue
import threading

import pygame,sys
SCREEN_WIDTH,SCREEN_HEIGHT= 800,680
BOARD_ORDER,BOARD_SIZE = 19,30
BOARD_X0,BOARD_Y0 = 15,15
GRID_NULL,GRID_BLACK,GRID_WHITE = 0,1,2
SPEED_X = [1,0,1,-1]
SPEED_Y = [0,1,1,1]
APP_TITLE = 'Gomoku AI'

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
        self.board.fill((240,200,0))
        L = BOARD_X0 + (BOARD_ORDER - 1) * BOARD_SIZE
        for i in range(BOARD_X0,SCREEN_HEIGHT,BOARD_SIZE):
            pygame.draw.line(self.board,(0,0,0),
                             (BOARD_X0,i),(L,i),2)
            pygame.draw.line(self.board,(0,0,0),
                             (i,BOARD_Y0),(i,L),1)
        pygame.draw.rect(self.board,(0,0,0),\
                         (BOARD_X0 - 1,BOARD_Y0 - 1,
                          L + 3 - BOARD_X0, L + 3 - BOARD_Y0),1)
        for star_y in (3,9,15):
            for star_x in (3,9,15):
                pygame.draw.circle(
                    self.board,
                    (0,0,0),
                    (
                        BOARD_X0 + star_x * BOARD_SIZE,
                        BOARD_Y0 + star_y * BOARD_SIZE,
                    ),
                    5,
                )
        return
    def __init__(self,bPic,wPic,x0,y0,ai_agent=None):
        self.bMan,self.wMan = bPic,wPic
        self.font = pygame.font.Font(None,32)
        self.fontScore = pygame.font.Font(None,16)
        self.fontMove = pygame.font.Font(None,18)
        self.fontWin = pygame.font.Font(None,96)
        self.fontTitle = pygame.font.Font(None,44)
        self.board = pygame.Surface((570,570))
        self.draw_board()
        self.x0,self.y0 = x0,y0
        self.ai_agent = ai_agent
        self.ai_results = queue.Queue()
        self.ai_thinking = False
        self.game_generation = 0
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
        self.black_button = pygame.Rect(625,300,145,40)
        self.white_button = pygame.Rect(625,350,145,40)
        self.undo_button = pygame.Rect(625,350,145,40)
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
        return
    def select_color(self,human_color):
        if human_color not in (GRID_BLACK,GRID_WHITE):
            return
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
        self.turn = self.human_color
        self.grid_assess()
        return True
    def draw_chess(self,scr):
        for y in range(BOARD_ORDER):
            for x in range(BOARD_ORDER):
                piece_color = self.grid[y][x]
                if self.grid[y][x] == GRID_BLACK:
                    scr.blit(self.bMan,\
                             (self.x0 + x * BOARD_SIZE,self.y0 + y * BOARD_SIZE))
                elif self.grid[y][x] == GRID_WHITE:
                    scr.blit(self.wMan,\
                             (self.x0 + x * BOARD_SIZE,self.y0 + y * BOARD_SIZE))
                elif self.assessFlag == True:
                    pnt = self.assessList[y][x]
                    if pnt.bAssess > 0 or pnt.bValue > 0:
                        txt = str(pnt.bAssess) + ',' + str(pnt.bValue)
                        RT_draw_txt(scr,self.fontScore,(0,0,0),txt,\
                                    self.x0 + x * BOARD_SIZE,self.y0 + y * BOARD_SIZE + 2)
                    if pnt.wAssess > 0 or pnt.wValue > 0:
                        txt = str(pnt.wAssess) + ',' + str(pnt.wValue)
                        RT_draw_txt(scr,self.fontScore,(255,255,255),txt,\
                                    self.x0 + x * BOARD_SIZE,self.y0 + y * BOARD_SIZE + 16)
                if piece_color != GRID_NULL:
                    piece_rect = pygame.Rect(
                        self.x0 + x * BOARD_SIZE,
                        self.y0 + y * BOARD_SIZE,
                        BOARD_SIZE,
                        BOARD_SIZE,
                    )
                    if (x,y) == self.last_ai_move:
                        pygame.draw.rect(scr,(235,0,180),piece_rect,2)
                    move_number = self.move_numbers[y][x]
                    if move_number:
                        number_color = (255,255,255) if piece_color == GRID_BLACK else (35,35,35)
                        number_pic = self.fontMove.render(str(move_number),True,number_color)
                        number_rect = number_pic.get_rect(center=piece_rect.center)
                        scr.blit(number_pic,number_rect)
        return
    def draw(self,scr):
        scr.fill((180,140,0))
        title_pic = self.fontTitle.render(APP_TITLE,True,(255,235,120))
        title_rect = title_pic.get_rect(center=(SCREEN_WIDTH // 2,36))
        scr.blit(title_pic,title_rect)
        scr.blit(self.board,(self.x0,self.y0))
        self.draw_chess(scr)
        RT_draw_txt(scr,self.fontScore,(255,245,200),'MODEL:',625,125)
        RT_draw_txt(scr,self.fontScore,(255,245,200),self.ai_model_label,625,145)
        if self.ai_search_label:
            RT_draw_txt(scr,self.fontScore,(255,220,80),self.ai_search_label,625,162)
        if self.human_color is None:
            status = 'CHOOSE COLOR'
            status_color = (255,220,80)
        elif self.sysStat == 1:
            status = 'GAME OVER'
            status_color = (245,245,245)
        elif self.ai_thinking:
            status = 'AI THINKING...'
            status_color = (255,220,80)
        else:
            status = 'YOUR TURN'
            status_color = (245,245,245)
        RT_draw_txt(scr,self.fontScore,status_color,status,625,185)
        RT_draw_txt(scr,self.fontScore,(245,235,190),'R/Enter: restart',625,225)
        RT_draw_txt(scr,self.fontScore,(245,235,190),'S: show scores',625,245)
        RT_draw_txt(scr,self.fontScore,(245,235,190),'C: change color',625,265)
        RT_draw_txt(scr,self.fontScore,(245,235,190),'U/Backspace: undo',625,285)
        if self.human_color is None:
            pygame.draw.rect(scr,(30,30,30),self.black_button)
            pygame.draw.rect(scr,(245,245,245),self.white_button)
            RT_draw_txt(scr,self.fontScore,(255,255,255),'B: BLACK (FIRST)',633,313)
            RT_draw_txt(scr,self.fontScore,(0,0,0),'W: WHITE (SECOND)',631,363)
        else:
            color_name = 'BLACK (FIRST)' if self.human_color == GRID_BLACK else 'WHITE (SECOND)'
            RT_draw_txt(scr,self.fontScore,(245,235,190),'YOU: ' + color_name,625,313)
            undo_color = (255,220,80) if self.can_undo() else (105,90,55)
            pygame.draw.rect(scr,undo_color,self.undo_button)
            RT_draw_txt(scr,self.fontScore,(25,25,25),'UNDO LAST TURN',638,363)
        if self.sysStat == 1:
            txt,cls = 'YOU WIN!!!',(255,0,0)
            if self.winner == self.ai_color:
                txt,cls = 'YOU LOSE!!!',(0,0,255)
            elif self.winner == GRID_NULL:
                txt,cls = 'DRAW',(80,80,80)
            RT_draw_txt(scr,self.fontWin,cls,txt,200,290)
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
                self.winner = GRID_NULL
                self.sysStat = 1
                return
            ai_x,ai_y = fallback
        self.grid[ai_y][ai_x] = self.ai_color
        self.move_count += 1
        self.move_numbers[ai_y][ai_x] = self.move_count
        self.move_history.append((ai_x,ai_y,self.ai_color))
        self.last_move = (ai_x,ai_y)
        self.last_ai_move = (ai_x,ai_y)
        if is_five(self.grid,ai_x,ai_y,self.ai_color):
            self.winner = self.ai_color
            self.sysStat = 1
            print('You lose!!!')
            return
        if board_full(self.grid):
            self.winner = GRID_NULL
            self.sysStat = 1
            return
        self.turn = self.human_color
        self.grid_assess()

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
        if self.sysStat == 1 or self.ai_thinking or self.turn != self.human_color:
            return
        x = round((mx - self.x0 - BOARD_X0) / BOARD_SIZE)
        y = round((my - self.y0 - BOARD_Y0) / BOARD_SIZE)
        if not (0 <= x < BOARD_ORDER and 0 <= y < BOARD_ORDER):
            return
        if self.grid[y][x] != GRID_NULL:
            return

        self.grid[y][x] = self.human_color
        self.move_count += 1
        self.move_numbers[y][x] = self.move_count
        self.move_history.append((x,y,self.human_color))
        self.last_move = (x,y)
        if is_five(self.grid, x, y, self.human_color):
            self.winner = self.human_color
            self.sysStat = 1
            print('You win!!!')
            return
        if board_full(self.grid):
            self.winner = GRID_NULL
            self.sysStat = 1
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
            self.grid_init(None)
            return
        if key in (pygame.K_u,pygame.K_BACKSPACE):
            self.undo()
            return
        if key in (pygame.K_RETURN,pygame.K_r):
            self.grid_init(self.human_color)
            if self.ai_color == GRID_BLACK:
                self._start_ai_move()
        if key == pygame.K_s:
            self.assessFlag = not self.assessFlag
        return


#-------main--------
def main():
    pygame.init()
    pygame.display.set_caption(APP_TITLE)
    screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
    asset_dir = Path(__file__).resolve().parent
    wPic = pygame.image.load(str(asset_dir / 'WCMan.bmp'))
    wPic.set_colorkey((255,0,0))
    bPic = pygame.image.load(str(asset_dir / 'BCMan.bmp'))
    bPic.set_colorkey((255,0,0))
    ai_agent = None
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
    gomoku = CLS_gomoku(bPic,wPic,30,80,ai_agent)
    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                gomoku.mouse_down(*event.pos)
            elif event.type == pygame.KEYDOWN:
                gomoku.eventkey(event.key)
            elif event.type == pygame.QUIT:
                running = False
        gomoku.update()
        gomoku.draw(screen)
        pygame.display.update()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == '__main__':
    sys.exit(main())
