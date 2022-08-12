import pygame

from pong.agent import UserAgent

from .environment import Ball, Environment, Player

white = (255, 255, 255)
gray = (128, 128, 128)
black = (0, 0, 0)
red = (255, 0, 0)


class Renderer(object):
    def __init__(self, screen_width, screen_height, env: Environment) -> None:
        pygame.init()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.user_agent = UserAgent()

        self.font = pygame.font.SysFont(None, 72)

        pygame.display.set_caption("Pong by Kinguuu")

        self.game_over = False
        self.env = env

        self.m2p = self.screen_width / (env.field.width - env.field.origin[0])

        self.clock = pygame.time.Clock()

    def draw_player(self, p: Player, upscale=1.0):
        x = upscale * p.pos_x
        y = upscale * p.pos_y
        w = upscale * p.width
        h = upscale * p.height
        pygame.draw.rect(self.screen, white, [x, y, w, h])

    def draw_ball(self, ball: Ball, upscale=1.0):
        pos = upscale * ball.pos
        rad = upscale * ball.radius
        pygame.draw.circle(self.screen, white, pos, rad)

    def draw_field(self, field, upscale=1.0):
        x_center = upscale * (field.origin[0] + 0.5 * field.width)
        self.screen.fill(black)
        pygame.draw.line(
            self.screen,
            gray,
            (x_center, upscale * field.origin[1]),
            (x_center, upscale * (field.origin[1] + field.height)),
        )

    def draw_score(self, upscale=1):
        score = self.env.states[-1]["score"].astype(int)
        img = self.font.render(f'{score[0]} : {score[1]}', True, gray, black)
        x_pos = self.env.field.origin[0] + (self.env.field.width - 0.25 * img.get_bounding_rect().width) * 0.5 * upscale
        y_pos = self.env.field.height * upscale * 0.05
        self.screen.blit(img, (x_pos, y_pos))

    def render(self, framerate: int = 0):
        self.draw_field(self.env.field, upscale=self.m2p)
        self.draw_player(self.env.p1, upscale=self.m2p)
        self.draw_player(self.env.p2, upscale=self.m2p)
        self.draw_score(self.m2p)
        self.draw_ball(self.env.ball, upscale=self.m2p)

        pygame.display.update()
        self.clock.tick(framerate)

    def events(self, user_control=False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if user_control:
                return self.user_agent.select_action(state=None, event=event)

        return self.user_agent.action

    def quit(self):
        pygame.quit()
