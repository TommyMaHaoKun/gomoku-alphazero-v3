import pygame,sys,time,random
SCREEN_WIDTH,SCREEN_HEIGHT= 800,680
BOARD_ORDER,BOARD_SIZE = 19,30
BOARD_X0,BOARD_Y0 = 15,15
GRID_NULL,GRID_BLACK,GRID_WHITE = 0,1,2
SPEED_X = []
SPEED_Y = []
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

class CLS_gomoku(object):
    def __init__(self,bPic,wPic,x0,y0):
        self.bMan,self.wMan = bPic,wPic
        self.x0,self.y0 = x0,y0
        self.board = pygame.Surface((570,570))
        self.draw_board()
        self.grid = []
        for y in range(BOARD_ORDER):
            line = [GRID_NULL] * BOARD_ORDER
            self.grid.append(line)
        self.flag = GRID_BLACK
        self.font = pygame.font.Font(None,36)
        self.fontTitle = pygame.font.Font(None,44)
        return
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
        return
    def draw(self,scr):
        scr.fill((180,140,0))
        title_pic = self.fontTitle.render(APP_TITLE,True,(255,235,120))
        title_rect = title_pic.get_rect(center=(SCREEN_WIDTH // 2,36))
        scr.blit(title_pic,title_rect)
        scr.blit(self.board,(self.x0,self.y0))
        for y in range(BOARD_ORDER):
            for x in range(BOARD_ORDER):
                if self.grid[y][x] == GRID_BLACK:
                    scr.blit(self.bMan,\
                             (self.x0 + x * BOARD_SIZE,self.y0 + y * BOARD_SIZE))
                elif self.grid[y][x] == GRID_WHITE:
                    scr.blit(self.wMan,\
                             (self.x0 + x * BOARD_SIZE,self.y0 + y * BOARD_SIZE))
        x = self.x0 + BOARD_X0 + BOARD_ORDER * BOARD_SIZE + 50
        txt = self.font.render('NEXT',True,(255,220,0))
        scr.blit(txt,(x,self.y0 + BOARD_Y0 + 20))
        if self.flag == GRID_BLACK:
            scr.blit(self.bMan,(x + 15,self.y0 + BOARD_Y0 + 50))
        else:
            scr.blit(self.wMan,(x + 15,self.y0 + BOARD_Y0 + 50))
        return
    def mouse_down(self,mx,my):
        gx = round((mx - 45) / 30)
        gy = round((my - 95) / 30)
        if 0 <= gx <BOARD_ORDER and 0 <= gy < BOARD_ORDER:
            if self.grid[gy][gx] == GRID_NULL:
                self.grid[gy][gx] = self.flag
                if is_five(self.grid,gx,gy,self.flag):
                    print(self.flag,'WIN!!!!')
                self.flag = GRID_BLACK + GRID_WHITE - self.flag
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


#-------main--------
pygame.init()
pygame.display.set_caption(APP_TITLE)
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
wPic = pygame.image.load('WCMan.bmp')
wPic.set_colorkey((255,0,0))
bPic = pygame.image.load('BCMan.bmp')
bPic.set_colorkey((255,0,0))
gomoku = CLS_gomoku(bPic,wPic,30,80)
while True:
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button != 1:
                continue
            mx,my = event.pos
            print(event.pos)
            gomoku.mouse_down(mx,my)
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    gomoku.draw(screen)
    pygame.display.update()
