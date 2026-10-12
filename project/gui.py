import os
import sys
import pygame

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "gin rummy"))

from play import Play

pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 1150, 720
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gin Rummy Desktop AI")

BG_DARK = (130, 15, 20)
BG_LIGHT = (195, 35, 40)
SLOT_BG = (90, 10, 15)
GOLD = (255, 215, 0)
WHITE = (255, 255, 255)
BTN_RED = (210, 35, 40)
BTN_BLUE = (35, 105, 215)

FONT_TITLE = pygame.font.SysFont("Trebuchet MS", 52, bold=True)
FONT_LARGE = pygame.font.SysFont("Trebuchet MS", 28, bold=True)
FONT_MED = pygame.font.SysFont("Trebuchet MS", 20, bold=True)
FONT_SMALL = pygame.font.SysFont("Trebuchet MS", 15, bold=True)

# Locate the spritesheet image dynamically across root directory
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "updated-card-2026.png"),
    os.path.join(BASE_DIR, "input_file_0.png"),
    os.path.join(BASE_DIR, "CuteCards - asset pack 0.1update", "CuteCards - asset pack", "CuteCards.png")
]

SPRITE_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        SPRITE_PATH = p
        break


class SpriteSheetLoader:
    """Robust image loader supporting both 840x480 grid (56x80px) and legacy sheets."""
    def __init__(self, path):
        self.sprites = {}
        if path and os.path.exists(path):
            sheet = pygame.image.load(path).convert_alpha()
            s_width, s_height = sheet.get_size()

            if s_width == 840 and s_height == 480:
                card_w, card_h = 56, 80
                row_map = {0: 'H', 1: 'S', 2: 'D', 3: 'C'}
                for row, suit in row_map.items():
                    for col in range(2, 15):
                        rank_num = col - 1  # 1 (Ace) to 13 (King)
                        key = f"{suit}{rank_num}"
                        rect = pygame.Rect(col * card_w, row * card_h, card_w, card_h)
                        surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                        surf.blit(sheet, (0, 0), rect)
                        self.sprites[key] = pygame.transform.smoothscale(surf, (75, 108))
                
                # Card Back (Red pattern at Col 1, Row 2)
                back_rect = pygame.Rect(1 * card_w, 2 * card_h, card_w, card_h)
                back_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                back_surf.blit(sheet, (0, 0), back_rect)
                self.card_back = pygame.transform.smoothscale(back_surf, (75, 108))
            else:
                card_w, card_h = 100, 144
                suit_map = {0: 'C', 1: 'D', 2: 'S', 3: 'H'}
                for row, suit in suit_map.items():
                    for col in range(13):
                        key = f"{suit}{col+1}"
                        rect = pygame.Rect(col * card_w, row * card_h, card_w, card_h)
                        surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                        surf.blit(sheet, (0, 0), rect)
                        self.sprites[key] = pygame.transform.smoothscale(surf, (75, 108))
                back_rect = pygame.Rect(14 * card_w, 2 * card_h, card_w, card_h)
                back_surf = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
                back_surf.blit(sheet, (0, 0), back_rect)
                self.card_back = pygame.transform.smoothscale(back_surf, (75, 108))
        else:
            self.card_back = pygame.Surface((75, 108))
            self.card_back.fill((45, 75, 160))

    def get_card(self, code: str):
        return self.sprites.get(code, self.card_back)


def draw_button(surface, text, rect, base_color, mouse_pos):
    color = (min(base_color[0]+30, 255), min(base_color[1]+30, 255), min(base_color[2]+30, 255)) if rect.collidepoint(mouse_pos) else base_color
    pygame.draw.rect(surface, color, rect, border_radius=10)
    pygame.draw.rect(surface, GOLD, rect, 2, border_radius=10)
    txt_surf = FONT_MED.render(text, True, WHITE)
    surface.blit(txt_surf, (rect.centerx - txt_surf.get_width() // 2, rect.centery - txt_surf.get_height() // 2))


def show_rules_dialog():
    pygame.event.clear()
    dialog_open = True
    while dialog_open:
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                dialog_open = False

        SCREEN.fill(BG_DARK)
        overlay = pygame.Rect(100, 80, WIDTH - 200, HEIGHT - 160)
        pygame.draw.rect(SCREEN, SLOT_BG, overlay, border_radius=16)
        pygame.draw.rect(SCREEN, GOLD, overlay, 3, border_radius=16)

        rules = [
            "GIN RUMMY RULES & SCORING",
            "1. Objective: Form Melds (Sets of 3/4 same rank or Runs of 3+ same suit).",
            "2. Draw a card each turn from Deck or Discard, then Discard one card.",
            "3. Knocking: Knock when Deadwood points are 10 or lower.",
            "4. Scoring System:",
            "   • Normal Knock: Difference in Deadwood (Knocker wins).",
            "   • Gin (0 Deadwood): Opponent Deadwood + 25 Bonus points.",
            "   • Undercut: Opponent has <= Deadwood. Opponent gains difference + 25 Bonus.",
            "5. First to 100 points wins the match!",
            "",
            "[ Click or Press Any Key to Return to Menu ]"
        ]

        for i, line in enumerate(rules):
            color = GOLD if i in [0, 4] else WHITE
            font = FONT_LARGE if i == 0 else FONT_MED
            t_surf = font.render(line, True, color)
            SCREEN.blit(t_surf, (140, 110 + i * 40))

        pygame.display.flip()
    pygame.event.clear()


def show_game_over_dialog(user_score: int, ai_score: int):
    pygame.event.clear()
    dialog_open = True
    user_won = user_score >= 100
    
    while dialog_open:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                dialog_open = False

        SCREEN.fill(BG_DARK)
        overlay = pygame.Rect(180, 120, WIDTH - 360, HEIGHT - 240)
        pygame.draw.rect(SCREEN, SLOT_BG, overlay, border_radius=16)
        
        border_color = GOLD if user_won else BTN_RED
        pygame.draw.rect(SCREEN, border_color, overlay, 3, border_radius=16)

        header_txt = "VICTORY!" if user_won else "DEFEAT!"
        header_color = GOLD if user_won else BTN_RED
        
        t_header = FONT_TITLE.render(header_txt, True, header_color)
        t_sub = FONT_LARGE.render("FINAL MATCH RESULTS", True, WHITE)
        t_user = FONT_LARGE.render(f"YOUR SCORE: {user_score}", True, GOLD)
        t_ai = FONT_LARGE.render(f"AI SCORE: {ai_score}", True, WHITE)
        t_prompt = FONT_MED.render("[ Click anywhere to return to Main Menu ]", True, GOLD)

        SCREEN.blit(t_header, (WIDTH // 2 - t_header.get_width() // 2, 160))
        SCREEN.blit(t_sub, (WIDTH // 2 - t_sub.get_width() // 2, 230))
        SCREEN.blit(t_user, (WIDTH // 2 - t_user.get_width() // 2, 310))
        SCREEN.blit(t_ai, (WIDTH // 2 - t_ai.get_width() // 2, 360))
        SCREEN.blit(t_prompt, (WIDTH // 2 - t_prompt.get_width() // 2, 460))

        pygame.display.flip()
    pygame.event.clear()


class GinRummyGUI:
    def __init__(self):
        self.loader = SpriteSheetLoader(SPRITE_PATH)
        self.game = Play()

    def main_menu(self):
        clock = pygame.time.Clock()
        in_menu = True

        while in_menu:
            m_pos = pygame.mouse.get_pos()
            SCREEN.fill(BG_DARK)
            pygame.draw.circle(SCREEN, BG_LIGHT, (WIDTH // 2, HEIGHT // 2), 380)

            title = FONT_TITLE.render("GIN RUMMY AI", True, GOLD)
            subtitle = FONT_MED.render("Ubisoft UNO-Inspired Desktop Suite", True, WHITE)
            SCREEN.blit(title, (WIDTH // 2 - title.get_width() // 2, 130))
            SCREEN.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 195))

            btn_start = pygame.Rect(WIDTH // 2 - 130, 310, 260, 60)
            btn_rules = pygame.Rect(WIDTH // 2 - 130, 400, 260, 60)

            draw_button(SCREEN, "START GAME", btn_start, BTN_RED, m_pos)
            draw_button(SCREEN, "RULES / HOW TO PLAY", btn_rules, BTN_BLUE, m_pos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_start.collidepoint(m_pos):
                        self.game = Play()
                        in_menu = False
                    elif btn_rules.collidepoint(m_pos):
                        show_rules_dialog()

            pygame.display.flip()
            clock.tick(30)

    def run_game_loop(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            m_pos = pygame.mouse.get_pos()
            SCREEN.fill(BG_DARK)
            pygame.draw.circle(SCREEN, BG_LIGHT, (WIDTH // 2, HEIGHT // 2), 420)

            # Action Buttons
            btn_surrender = pygame.Rect(30, HEIGHT - 70, 140, 45)
            btn_sort = pygame.Rect(WIDTH - 180, 15, 150, 35)
            btn_knock = pygame.Rect(WIDTH - 170, HEIGHT - 70, 140, 45)
            btn_next = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 70, 200, 45)

            draw_button(SCREEN, "SURRENDER", btn_surrender, BTN_RED, m_pos)
            draw_button(SCREEN, f"SORT: {self.game.user_sort_mode}", btn_sort, BTN_BLUE, m_pos)

            user_dw = self.game.user_hand.calculate_deadwood()
            
            if self.game.current_turn == "USER" and user_dw <= 10 and not self.game.round_over:
                btn_label = "GIN!" if user_dw == 0 else "KNOCK"
                btn_color = (220, 160, 20) if user_dw == 0 else BTN_BLUE
                draw_button(SCREEN, btn_label, btn_knock, btn_color, m_pos)

            if self.game.round_over:
                draw_button(SCREEN, "NEXT ROUND", btn_next, BTN_BLUE, m_pos)

            # Table Slots
            deck_rect = pygame.Rect(WIDTH // 2 - 90, HEIGHT // 2 - 54, 75, 108)
            disc_rect = pygame.Rect(WIDTH // 2 + 15, HEIGHT // 2 - 54, 75, 108)
            pygame.draw.rect(SCREEN, SLOT_BG, deck_rect, border_radius=8)
            pygame.draw.rect(SCREEN, SLOT_BG, disc_rect, border_radius=8)

            if self.game.deck_remaining > 0:
                SCREEN.blit(self.loader.card_back, deck_rect.topleft)
            top_disc = self.game.discard_pile.top()
            if top_disc:
                SCREEN.blit(self.loader.get_card(top_disc), disc_rect.topleft)

            # AI Hand
            ai_cards = self.game.ai_hand.get_cards()
            start_x_ai = WIDTH // 2 - (len(ai_cards) * 22)
            for i, code in enumerate(ai_cards):
                card_img = self.loader.get_card(code) if self.game.round_over else self.loader.card_back
                SCREEN.blit(card_img, (start_x_ai + i * 44, 35))

            # Player Hand
            cards = self.game.user_hand.sort_cards_by_mode(self.game.user_sort_mode)
            card_rects = []
            start_x = WIDTH // 2 - (len(cards) * 26)
            for i, code in enumerate(cards):
                rect = pygame.Rect(start_x + i * 52, HEIGHT - 180, 75, 108)
                SCREEN.blit(self.loader.get_card(code), rect.topleft)
                card_rects.append((rect, code))

            # HUD Display
            hud = FONT_MED.render(
                f"YOU: {self.game.user_score} PTS | AI: {self.game.ai_score} PTS | AI Acc: {self.game.ai_engine.accuracy:.1f}%", 
                True, GOLD
            )
            dw_txt = FONT_SMALL.render(f"Your Deadwood: {user_dw}", True, WHITE)
            msg = FONT_MED.render(self.game.status_msg, True, WHITE)

            SCREEN.blit(hud, (30, 15))
            SCREEN.blit(dw_txt, (30, 42))
            SCREEN.blit(msg, (WIDTH // 2 - msg.get_width() // 2, HEIGHT - 220))

            pygame.display.flip()

            if self.game.round_over and (self.game.user_score >= 100 or self.game.ai_score >= 100):
                pygame.time.delay(1000)
                show_game_over_dialog(self.game.user_score, self.game.ai_score)
                return

            if self.game.current_turn == "AI" and not self.game.round_over:
                pygame.time.delay(600)
                self.game.execute_ai_turn()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if btn_sort.collidepoint(m_pos):
                        self.game.cycle_sort_mode()
                    elif btn_surrender.collidepoint(m_pos):
                        self.game.handle_surrender()
                    elif self.game.round_over and btn_next.collidepoint(m_pos):
                        self.game.start_new_round()
                    elif self.game.current_turn == "USER" and not self.game.round_over:
                        if btn_knock.collidepoint(m_pos) and user_dw <= 10:
                            self.game.handle_user_knock()
                        elif self.game.current_phase == "DRAW":
                            if deck_rect.collidepoint(m_pos):
                                self.game.handle_user_draw_deck()
                            elif disc_rect.collidepoint(m_pos):
                                self.game.handle_user_draw_discard()
                        elif self.game.current_phase == "DISCARD":
                            for rect, code in card_rects:
                                if rect.collidepoint(m_pos):
                                    self.game.handle_user_discard(code)
                                    break
            clock.tick(30)
        pygame.quit()

    def run(self):
        while True:
            self.main_menu()
            self.run_game_loop()
