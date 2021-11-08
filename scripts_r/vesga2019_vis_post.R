library(tidyverse)
library(tidybayes)


#theme_set(theme_bw() + theme(text=element_text(family="serif")))
theme_set(theme_tidybayes() + theme(text=element_text(family="sans")))




sims_ctrl <- read_csv("out/Vesga2019/Runs_Ctrl.csv") %>% mutate(Scenario = "Base")
sims_intv <- read_csv("out/Vesga2019/Runs_Intv.csv") %>% mutate(Scenario = "Intv")


targets <- read_csv("data/Vesga2019/Targets.csv")


sims <- bind_rows(sims_ctrl, sims_intv) %>% 
  select(Time, Scenario, Prevalence = PrevTB, Incidence = IncTB, Mortality = MorTB) %>% 
  group_by(Time, Scenario) %>% 
  summarise(across(c(Prevalence, Incidence, Mortality), list(
    M = mean,
    L = function(x) quantile(x, 0.025),
    U = function(x) quantile(x, 0.975)
  ))) %>% 
  pivot_longer(c(-Time, -Scenario)) %>% 
  separate(name, c("Index", "Stats"), "_") %>% 
  pivot_wider(c(Time, Scenario, Index), names_from = Stats, values_from = value) %>% 
  mutate(
    Index = factor(Index, c("Prevalence", "Incidence", "Mortality"))
  )

dat <- targets %>% 
  filter(Group == "Epi") %>% 
  mutate(
    Index = factor(Index, c("Prevalence", "Incidence", "Mortality"))
  )


g_epi <- sims %>% 
  filter(Time > 2015) %>% 
  ggplot(aes(x = Time)) +
  geom_ribbon(aes(ymin = L, ymax = U, fill = Scenario), alpha = 0.2) +
  geom_line(aes(y = M, colour = Scenario)) +
  geom_pointinterval(data = dat, aes(y = M, ymin = L, ymax = U)) +
  scale_y_continuous("Per 100 000", labels = scales::label_number(scale = 1E5)) +
  scale_x_continuous("Year", breaks = seq(2015, 2035, 2)) +
  scale_colour_hue("Scenario", labels = c(Base = "Baseline", Intv = "Intervention")) +
  scale_fill_hue("Scenario", labels = c(Base = "Baseline", Intv = "Intervention")) +
  facet_wrap(.~Index, ncol = 1, scales = "free_y") +
  expand_limits(y = 0)


g_epi




sims <- sims_ctrl %>% 
  select(Time, Scenario, starts_with("Pr"), - PrevTB) %>% 
  group_by(Time, Scenario) %>% 
  summarise(across(starts_with("Pr"), list(
    M = mean,
    L = function(x) quantile(x, 0.025),
    U = function(x) quantile(x, 0.975)
  ))) %>% 
  pivot_longer(c(-Time, -Scenario)) %>% 
  separate(name, c("Index", "Stats"), "_") %>% 
  pivot_wider(c(Time, Scenario, Index), names_from = Stats, values_from = value) %>% 
  mutate(
    Index = factor(Index, c("PrAsym", "PrSym", "PrCS", "PrTxPri", "PrTxPub")),
    Type = "Simulation"
  ) %>% 
  filter(Time ==2016)

dat <- targets %>% 
  filter(Group == "Frac") %>% 
  mutate(
    Index = factor(Index, c("PrAsym", "PrSym", "PrCS", "PrTxPri", "PrTxPub")),
    Type = "Data"
  )


g_frac <- bind_rows(sims, dat) %>% 
  ggplot() +
  geom_pointrange(aes(x = Index, y = M, ymin = L, ymax = U, shape = Type), 
                  position = position_dodge2(0.3)) +
  scale_y_continuous("Fraction, %", labels = scales::percent) +
  scale_x_discrete("Prevalent TB") +
  expand_limits(y = c(0, 0.5)) +
  labs(title = "Gujarat TB survey, 2016")

g_frac



ggsave(g_frac, filename = here::here("docs", "Vesga2019", "Post_FracPrev.png"), width = 6, height = 4)
ggsave(g_epi, filename = here::here("docs", "Vesga2019", "Post_Epi.png"), width = 6, height = 4)




