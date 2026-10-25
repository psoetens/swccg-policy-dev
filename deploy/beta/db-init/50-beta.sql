-- Plan beta-server: runs once, on the first start of an empty database, after Gemp's own scripts (10-40).
-- Gemp's seeded users have known passwords: all go except Librarian (owner of the sample decks), whose password
-- becomes a random hash nobody knows. Bot tables on (Gemp's default is off: "Bot tables are currently disabled"),
-- registration on (plan decision B1; the admin switches it in Admin -> settings).
DELETE FROM `gemp-swccg`.`player` WHERE name <> 'Librarian';
UPDATE `gemp-swccg`.`player` SET password = SHA2(CONCAT(UUID(), RAND()), 256) WHERE name = 'Librarian';
UPDATE `gemp-swccg`.`gemp_settings` SET settingValue = 1 WHERE settingName IN ('aiTablesEnabled', 'newAccountRegistrationEnabled');
